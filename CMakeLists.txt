# CMakeList.txt : projet CMake pour PComNorm, incluez la source et définissez
# la logique spécifique au projet ici.
#
cmake_minimum_required (VERSION 3.8)

# Activez Rechargement à chaud pour les compilateurs MSVC si cela est pris en charge.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()
include(D:/vcpkg/scripts/buildsystems/vcpkg.cmake)
project ("PComNorm")
include(D:/vcpkg/scripts/buildsystems/vcpkg.cmake)
# Ajoutez une source à l'exécutable de ce projet.
add_executable (PComNorm "PComNorm.cpp" )
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(stb REQUIRED)
find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries (PComNorm OpenGL::GL GLEW::GLEW glfw glm::glm ${Stb_INCLUDE_DIR} ${OpenCV_LIBS} )
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET PComNorm PROPERTY CXX_STANDARD 20)
endif()

# TODO: Ajoutez des tests et installez des cibles si nécessaire.
