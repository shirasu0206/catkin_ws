execute_process(COMMAND "/home/syu/catkin_ws/build/robot_helpers/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/syu/catkin_ws/build/robot_helpers/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
