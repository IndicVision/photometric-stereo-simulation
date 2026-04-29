cmake_minimum_required(VERSION 3.4.1)

# 1. The OpenCV map
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../sdk/native/jni)
find_package(OpenCV REQUIRED)

# 2. Gather all the LibRaw C++ source files
file(GLOB_RECURSE LIBRAW_SOURCES "libraw/src/*.cpp")

# --- THE FIX: Throw away the duplicate aggregator files! ---
file(GLOB_RECURSE LIBRAW_DUPLICATES "libraw/src/*_ph.cpp")
if(LIBRAW_DUPLICATES)
    list(REMOVE_ITEM LIBRAW_SOURCES ${LIBRAW_DUPLICATES})
endif()
# -----------------------------------------------------------

# 3. Build your engine
add_library( ps_engine SHARED ps_engine.cpp ${LIBRAW_SOURCES} )

# 4. Include LibRaw headers
target_include_directories( ps_engine PRIVATE libraw )

# 5. Disable extra LibRaw features (saves space)
target_compile_definitions( ps_engine PRIVATE -DNO_JPEG -DNO_JASPER -DNO_ZLIB )

# 6. Link everything together
find_library( log-lib log )
target_link_libraries( ps_engine ${log-lib} ${OpenCV_LIBS} )