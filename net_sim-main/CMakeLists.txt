cmake_minimum_required(VERSION 3.17)
project(net_sim)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall -Wextra -Werror -Wpedantic -pedantic-errors -Werror=switch)

include_directories(
        include
)

set(SOURCE_FILES
        src/factory.cpp
        src/helpers.cpp
        src/nodes.cpp
        src/package.cpp
        src/reports.cpp
        src/simulation.cpp
        src/storage_types.cpp
        )

#
add_executable(net_sim ${SOURCE_FILES} main.cpp)

set(SOURCE_FILES_TESTS
        netsim_tests/test/test_Factory.cpp
        netsim_tests/test/test_factory_io.cpp
        netsim_tests/test/test_nodes.cpp
        netsim_tests/test/test_package.cpp
        netsim_tests/test/test_reports.cpp
        netsim_tests/test/test_simulate.cpp
        netsim_tests/test/test_storage_types.cpp
        src/simulation.cpp include/simulation.hpp)

add_executable(netsim_tests ${SOURCE_FILES} ${SOURCE_FILES_TESTS} netsim_tests/test/main_gtest.cpp)

target_include_directories(netsim_tests
        PUBLIC
        googletest-master/googlemock/include
        googletest-master/googletest/include
        mocks
        netsim_tests/mocks

        )

add_subdirectory(googletest-master)

target_link_libraries(netsim_tests gmock)

#target_compile_definitions(netsim_test PUBLIC EXERCISE_ID=1)
