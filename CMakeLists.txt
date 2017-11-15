cmake_minimum_required(VERSION 2.8)
project(pnp)

set(CMAKE_CXX_STANDARD 11)


IF(CMAKE_COMPILER_IS_GNUCC)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++0x -fext-numeric-literals")
ELSE()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF()


INCLUDE(ExternalProject)
LIST(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

FIND_PACKAGE(G2O REQUIRED QUIET)
MESSAGE("-- find G2O")
FIND_PACKAGE(SuiteSparse REQUIRED)
MESSAGE("-- find SuiteSparse")
FIND_PACKAGE(Glog REQUIRED QUIET)
MESSAGE("-- find Glog")
FIND_PACKAGE(Cholmod REQUIRED QUIET)
MESSAGE("-- find Cholmod")
FIND_PACKAGE(BLAS REQUIRED QUIET)
MESSAGE("-- find blas")
FIND_PACKAGE(LAPACK REQUIRED QUIET)
MESSAGE("-- find lapack")
FIND_PACKAGE(Ceres REQUIRED QUIET)
MESSAGE("-- find Ceres")
FIND_PACKAGE(FREEGLUT REQUIRED QUIET)
MESSAGE("-- find freeglut")
FIND_PACKAGE(GLEW REQUIRED QUIET)
MESSAGE("-- find glew")
FIND_PACKAGE(Boost REQUIRED COMPONENTS thread random math_c99)

FIND_LIBRARY(GOMP_LIB
        NAMES gomp libgomp.so.1
        PATH
        /usr/lib/x86_64-linux-gnu
        /usr/lib
        /usr/local/lib
        )

set(SOURCE_FILES
        main.cpp
        PNPSimulation.cpp
        PNPSimulation.h
        )


INCLUDE_DIRECTORIES(
        .
        ./Thirdparty
        ${GLOG_INCLUDE_DIR}
        ${CERES_INCLUDE_DIR}
        ${SuiteSparse_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIR}
        ${G2O_INCLUDE_DIRS}
)



FIND_LIBRARY(GLOG_LIB
        NAMES glog glog.so libglog.so
        PATH
        /usr/lib/x86_64-linux-gnu
        /opt/local/lib
        /usr/lib
        /usr/local/lib
        )

LIST(APPEND LINK_LIBS
        ${GLOG_LIBRARIES}
        ${CERES_LIBRARIES}
        ${QGLVIEWER_LIBRARY_RELEASE}
        ${SUITESPARSE_LIBRARIES}
        ${CHOLMOD_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${GLOG_LIB}
        ${G2O_CORE_LIBRARY}
        ${G2O_STUFF_LIBRARY}
        ${GOMP_LIB}
        )



add_executable(pnp ${SOURCE_FILES} ${UI_SOURCES})
TARGET_LINK_LIBRARIES(pnp ${LINK_LIBS} pthread)
