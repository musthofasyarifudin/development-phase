cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(app_pred)

set(CMAKE_PREFIX_PATH "E:/open_project/Delameta_/libtorch-shared-with-deps-latest/libtorch/")

find_package(Torch REQUIRED)

add_executable(app_pred app_pred.cpp)

target_link_libraries(app_pred "${TORCH_LIBRARIES}")

set_property(TARGET app_pred PROPERTY CXX_STANDARD 14)

if (MSVC)
  file(GLOB TORCH_DLLS "E:/open_project/Delameta_/libtorch-shared-with-deps-latest/libtorch/lib/*.dll")
  add_custom_command(TARGET app_pred
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     $<{TORCH_DLLS}: app_pred>)
endif (MSVC)
