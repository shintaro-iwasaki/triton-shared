
# Python module
if(TRITON_BUILD_PYTHON_MODULE)
    message(STATUS "Adding Triton-Shared Reference CPU Backend")
    file(INSTALL
         ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
         DESTINATION ${PYTHON_THIRD_PARTY_PATH}/cpu/)
    # TODO: perhaps we want to install binary files used in __init__.py
endif()
