
XCode issues

- Had to set compiler to LLVM-GCC 4.2 inside of XCode

OpenCV 2.4.3 issues / workarounds

- Had to build OpenCV 2.4.3 from source in order to enable CUDA
- Had to compile with FFMPEG turned off
- Not sure why FFMPEG isn't compiling properly

- Used these flags to set build packages:
  > cmake -DWITH_QT=YES -DWITH_CUDA=YES -DWITH_FFMPEG=OFF -DWITH_OPENGL=ON
  > make
  > sudo make install
  
- Trying to build FFMPEG via MacPorts with some extra flags (hopefully this will allow me to use -DWITH_FFMPEG_ON):
    > sudo port install ffmpeg +gpl +postproc +lame +theora +libogg +vorbis +xvid +x264 +a52 +faac +faad +dts +nonfree

- Had to set gcc as the CUDA compiler instead of cc (which is an alias for clang) - see http://makeclean.iobloggo.com/460/opencv-243-with-cuda-on-osx-with-xcode-452/&cid=6090

- To force OpenCV to compile against OpenGL had to replace line 128 of CMakeList.txt to read:
  OCV_OPTION(WITH_OPENGL         "Include OpenGL support"                      ON )

- Used MacPorts to install QT4 : sudo port install qt4-mac +universal +debug
(//this may not be necessary if we have FFMPEG built properly - see below)


- Had to comment out reference to glXUseXFont which is not supported by Apples GL or AGL framework in this file: /Users/angus/OpenCV-2.4.3/modules/highgui/src/window_QT.cpp

FFMPEG issues

- Trying to build FFMPEG via MacPorts with some extra flags (hopefully this will allow me to use -DWITH_FFMPEG_ON):
    > sudo port install ffmpeg +gpl +postproc +lame +theora +libogg +vorbis +xvid +x264 +a52 +faac +faad +dts +nonfree

