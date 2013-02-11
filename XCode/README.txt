
- Had to set compiler to LLVM-GCC 4.2 inside of XCode
- Had to build from source in order to enable CUDA
- Had to compile with FFMPEG turned off

- Used these flags to set build packages:
  cmake -DWITH_QT=YES -DWITH_CUDA=YES -DWITH_FFMPEG=OFF -DWITH_OPENGL=ON
  
- Not sure why FFMPEG isn't compiling properly

- Had to set gcc as the CUDA compiler instead of cc (which is an alias for clang) - see http://makeclean.iobloggo.com/460/opencv-243-with-cuda-on-osx-with-xcode-452/&cid=6090

- To force OpenCV to compile against OpenGL had to replace line 128 of CMakeList.txt to read:
  OCV_OPTION(WITH_OPENGL         "Include OpenGL support"                      ON )

- Used MacPorts to install QT4 : sudo port install qt4-mac +universal +debug

- Had to comment out reference to glXUseXFont which is not supported by Apples GL or AGL framework





