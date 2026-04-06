gradation.png(要librsvg):
```
ffmpeg -width 3200 -i gradation.svg gradation.png
```

gradation-fs8.png:
```
pngquant 256 .\gradation.png
```
gradation-256.png:
```
magick gradation.png PNG8:gradation-256.png
```
