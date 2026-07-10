//! Windows video decoding through a D3D11VA-only FFmpeg path.
//!
//! The build intentionally contains only the demuxers and decoders selected
//! in `tools/build_ffmpeg_minimal.ps1`. A decoded frame must be a D3D11 frame;
//! a CPU decoder result is rejected instead of silently falling back.

use dssim_core::ToRGBAPLU;
use ffmpeg::ffi;
use ffmpeg::util::error::EAGAIN;
use ffmpeg_next as ffmpeg;
use imgref::Img;
use std::collections::VecDeque;
use std::path::Path;
use std::ptr;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const VIDEO_EXTENSIONS: &[&str] = &["mp4", "m4v", "mov", "mkv", "webm"];

/// Returns true only for containers compiled into the minimal FFmpeg build.
pub fn is_video_path(path: impl AsRef<Path>) -> bool {
    path.as_ref()
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            VIDEO_EXTENSIONS
                .iter()
                .any(|known| ext.eq_ignore_ascii_case(known))
        })
}

/// Aggregate produced when two videos are compared frame-by-frame.
#[derive(Clone, Copy, Debug)]
pub struct VideoComparison {
    pub score: f64,
    pub frames: usize,
}

/// Compares corresponding decoded frames and returns their arithmetic mean DSSIM.
///
/// Both videos must contain the same number of video frames and every pair must
/// have the same dimensions. Audio and subtitles are deliberately ignored.
pub fn compare_video_files(
    attr: &crate::Dssim,
    original_path: impl AsRef<Path>,
    modified_path: impl AsRef<Path>,
) -> Result<VideoComparison> {
    ffmpeg::init()?;

    let mut original = VideoReader::open(original_path.as_ref())?;
    let mut modified = VideoReader::open(modified_path.as_ref())?;
    let mut total = 0.0_f64;
    let mut frames = 0_usize;

    loop {
        let left = original.next_image(attr)?;
        let right = modified.next_image(attr)?;

        match (left, right) {
            (None, None) => break,
            (None, Some(_)) | (Some(_), None) => {
                return Err("Videos have different decoded frame counts".into());
            }
            (Some(left), Some(right)) => {
                if left.width() != right.width() || left.height() != right.height() {
                    return Err(format!(
                        "Video frame {frames} has different dimensions ({}x{} vs {}x{})",
                        left.width(),
                        left.height(),
                        right.width(),
                        right.height(),
                    )
                    .into());
                }
                total += f64::from(attr.compare(&left, right).0);
                frames += 1;
            }
        }
    }

    if frames == 0 {
        return Err("No decodable video frames were found".into());
    }

    Ok(VideoComparison {
        score: total / frames as f64,
        frames,
    })
}

struct HardwareDevice(*mut ffi::AVBufferRef);

impl HardwareDevice {
    fn d3d11va() -> Result<Self> {
        let ty = unsafe { ffi::av_hwdevice_find_type_by_name(b"d3d11va\0".as_ptr().cast()) };
        if ty == ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE {
            return Err("This FFmpeg build does not include D3D11VA".into());
        }

        let mut device = ptr::null_mut();
        let result = unsafe {
            ffi::av_hwdevice_ctx_create(&mut device, ty, ptr::null(), ptr::null_mut(), 0)
        };
        if result < 0 {
            return Err(ffmpeg::Error::from(result).into());
        }
        Ok(Self(device))
    }

    fn as_ptr(&self) -> *mut ffi::AVBufferRef {
        self.0
    }
}

impl Drop for HardwareDevice {
    fn drop(&mut self) {
        unsafe { ffi::av_buffer_unref(&mut self.0) };
    }
}

/// Fail before opening the decoder if the linked `avcodec` does not expose the
/// D3D11VA2 configuration that backs `AV_PIX_FMT_D3D11`.
///
/// Merely seeing `D3D11` in a codec's `get_format` candidates is insufficient:
/// that list can be present even when the corresponding hwaccel object was
/// left out of a slim static build.
fn require_d3d11va2_configuration(codec: &ffmpeg::codec::codec::Codec) -> Result<()> {
    let mut configurations = Vec::new();
    let required_method = ffi::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32;

    for index in 0.. {
        let configuration = unsafe { ffi::avcodec_get_hw_config(codec.as_ptr(), index) };
        if configuration.is_null() {
            break;
        }

        let configuration = unsafe { &*configuration };
        let pixel_format = ffmpeg::format::Pixel::from(configuration.pix_fmt);
        configurations.push(format!(
            "{pixel_format:?} (methods=0x{:x}, device={:?})",
            configuration.methods, configuration.device_type
        ));

        if pixel_format == ffmpeg::format::Pixel::D3D11
            && configuration.device_type == ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA
            && configuration.methods & required_method != 0
        {
            return Ok(());
        }
    }

    Err(format!(
        "The linked FFmpeg decoder '{}' has no D3D11VA device configuration. Available hardware configurations: {}",
        codec.name(),
        if configurations.is_empty() {
            "none".to_owned()
        } else {
            configurations.join(", ")
        }
    )
    .into())
}

unsafe extern "C" fn select_d3d11_format(
    _context: *mut ffi::AVCodecContext,
    formats: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    let mut format = formats;
    while !format.is_null() {
        let candidate = unsafe { *format };
        if ffmpeg::format::Pixel::from(candidate) == ffmpeg::format::Pixel::D3D11 {
            return candidate;
        }
        if ffmpeg::format::Pixel::from(candidate) == ffmpeg::format::Pixel::None {
            break;
        }
        format = unsafe { format.add(1) };
    }
    ffmpeg::format::Pixel::None.into()
}

struct VideoReader {
    input: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Video,
    stream_index: usize,
    // Keep this alive until after the decoder is dropped.
    _hardware_device: HardwareDevice,
    frames: VecDeque<ffmpeg::frame::Video>,
    sent_eof: bool,
    decoder_eof: bool,
}

impl VideoReader {
    fn open(path: &Path) -> Result<Self> {
        let input = ffmpeg::format::input(path)?;
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or("No video stream was found")?;
        let stream_index = stream.index();
        let codec = ffmpeg::codec::decoder::find(stream.parameters().id())
            .ok_or("The video codec is not enabled in the minimal FFmpeg build")?;
        require_d3d11va2_configuration(&codec)?;

        let hardware_device = HardwareDevice::d3d11va()?;
        let mut context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;
        unsafe {
            let raw_context = context.as_mut_ptr();
            (*raw_context).get_format = Some(select_d3d11_format);
            (*raw_context).extra_hw_frames = 32;
            (*raw_context).hw_device_ctx = ffi::av_buffer_ref(hardware_device.as_ptr());
            if (*raw_context).hw_device_ctx.is_null() {
                return Err("Failed to attach the D3D11VA device to the decoder".into());
            }
        }

        let decoder = context.decoder().open_as(codec)?.video()?;
        Ok(Self {
            input,
            decoder,
            stream_index,
            _hardware_device: hardware_device,
            frames: VecDeque::new(),
            sent_eof: false,
            decoder_eof: false,
        })
    }

    fn next_image(&mut self, attr: &crate::Dssim) -> Result<Option<crate::DssimImage<f32>>> {
        loop {
            if let Some(frame) = self.frames.pop_front() {
                return rgb_frame_to_dssim(attr, &frame).map(Some);
            }
            if self.decoder_eof {
                return Ok(None);
            }

            self.drain_decoder()?;
            if !self.frames.is_empty() || self.decoder_eof {
                continue;
            }

            let packet = self
                .input
                .packets()
                .next()
                .map(|(stream, packet)| (stream.index(), packet));
            match packet {
                Some((index, packet)) if index == self.stream_index => {
                    self.decoder.send_packet(&packet)?;
                }
                Some(_) => {}
                None if !self.sent_eof => {
                    self.decoder.send_eof()?;
                    self.sent_eof = true;
                }
                None => return Ok(None),
            }
        }
    }

    fn drain_decoder(&mut self) -> Result<()> {
        loop {
            let mut decoded = ffmpeg::frame::Video::empty();
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => self.frames.push_back(copy_d3d11_frame_to_rgb(&decoded)?),
                Err(ffmpeg::Error::Other { errno }) if errno == EAGAIN => return Ok(()),
                Err(ffmpeg::Error::Eof) => {
                    self.decoder_eof = true;
                    return Ok(());
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
}

fn copy_d3d11_frame_to_rgb(decoded: &ffmpeg::frame::Video) -> Result<ffmpeg::frame::Video> {
    if decoded.format() != ffmpeg::format::Pixel::D3D11 {
        return Err("FFmpeg selected a software decoder; hardware decoding is required".into());
    }

    let mut system_frame = ffmpeg::frame::Video::empty();
    let result =
        unsafe { ffi::av_hwframe_transfer_data(system_frame.as_mut_ptr(), decoded.as_ptr(), 0) };
    if result < 0 {
        return Err(ffmpeg::Error::from(result).into());
    }

    let mut rgb_frame = ffmpeg::frame::Video::empty();
    let mut scaler = ffmpeg::software::scaling::Context::get(
        system_frame.format(),
        system_frame.width(),
        system_frame.height(),
        ffmpeg::format::Pixel::RGB24,
        system_frame.width(),
        system_frame.height(),
        ffmpeg::software::scaling::flag::Flags::BILINEAR,
    )?;
    scaler.run(&system_frame, &mut rgb_frame)?;
    Ok(rgb_frame)
}

fn rgb_frame_to_dssim(
    attr: &crate::Dssim,
    frame: &ffmpeg::frame::Video,
) -> Result<crate::DssimImage<f32>> {
    if frame.format() != ffmpeg::format::Pixel::RGB24 {
        return Err("Internal error: expected an RGB24 video frame".into());
    }
    let width = frame.width() as usize;
    let height = frame.height() as usize;
    let row_bytes = width.checked_mul(3).ok_or("Video width is too large")?;
    let stride = frame.stride(0);
    let source = frame.data(0);
    let mut pixels = Vec::with_capacity(
        width
            .checked_mul(height)
            .ok_or("Video frame is too large")?,
    );

    for row in 0..height {
        let start = row
            .checked_mul(stride)
            .ok_or("Video frame stride overflow")?;
        let end = start
            .checked_add(row_bytes)
            .ok_or("Video frame stride overflow")?;
        let bytes = source
            .get(start..end)
            .ok_or("Invalid RGB video frame stride")?;
        pixels.extend(
            bytes
                .chunks_exact(3)
                .map(|pixel| rgb::RGB8::new(pixel[0], pixel[1], pixel[2])),
        );
    }

    let image = Img::new(pixels.to_rgblu(), width, height);
    attr.create_image(&image)
        .ok_or_else(|| "FFmpeg produced an empty video frame".into())
}
