#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "LowRankKMeans::lowrank_kmeans" for configuration "Debug"
set_property(TARGET LowRankKMeans::lowrank_kmeans APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(LowRankKMeans::lowrank_kmeans PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib64/liblowrank_kmeans.a"
  )

list(APPEND _cmake_import_check_targets LowRankKMeans::lowrank_kmeans )
list(APPEND _cmake_import_check_files_for_LowRankKMeans::lowrank_kmeans "${_IMPORT_PREFIX}/lib64/liblowrank_kmeans.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
