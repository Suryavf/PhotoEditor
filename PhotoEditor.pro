#-------------------------------------------------
#
# Project created by QtCreator 2018-11-14T23:28:40
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Aplication
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

SOURCES += \
        main.cpp \
        photoeditor.cpp \
        library/bmp.cpp \
        Filter/Filter.cpp \
        GeomTrans/geomTransform.cpp

HEADERS += \
        photoeditor.h \
        photoeditor.h \
        library/bmp.h \
        ColorSpace/ColorSpace.h \
        Filter/Filter.h \
        Filter/kernel.h \
        Filter/Constants.h \
        FFT/fft.h \
        GeomTrans/geomTransform.h

# Cuda files:
OTHER_FILES += \
        ColorSpace/ColorSpace.cu \
        FFT/fft.cu

CUDA_SOURCES += ColorSpace/ColorSpace.cu \
                FFT/fft.cu
CUDA_SDK = "/usr/local/cuda/"   # Path to cuda SDK install
CUDA_DIR = "/usr/local/cuda/"

# Cuda Options:

SYSTEM_NAME = x64         # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64           # '32' or '64', depending on your system
CUDA_ARCH = sm_30          # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS = --use_fast_math

# include paths
INCLUDEPATH += $$CUDA_DIR/include
INCLUDEPATH += /usr/local/include/opencv2

# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib/

CUDA_OBJECTS_DIR = ./

# Add the necessary libraries
CUDA_LIBS = -lcuda -lcudart -lcufft

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
#LIBS += $$join(CUDA_LIBS,'.so ', '', '.so')
LIBS += $$CUDA_LIBS

# OpenCV
LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio
CONFIG += link_pkgconfig
PKGCONFIG += opencv

# OpenMP
QMAKE_CXXFLAGS += -fopenmp
LIBS += -fopenmp

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input    = CUDA_SOURCES
    cuda_d.output   = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input    = CUDA_SOURCES
    cuda.output   = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}

FORMS += \
        photoeditor.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    ColorSpace/ColorSpace.cu \
    FFT/fft.cu
