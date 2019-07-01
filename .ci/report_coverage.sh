#! /bin/bash -e

# Only collect and upload coverage reports from C++17 Builds
# This is required because Tensor Extension only runs with C++17

# Capture all the coverge reports
lcov --rc lcov_branch_coverage=1 --directory ${TLIB_ROOT} --capture --output-file coverage.info 

#--no-external 
    
# Remove all unwanted coverages libs. 
# Boost.uBLAS depends uses many internal boost libs, we don't want them to be in coverage.
    
#lcov --rc lcov_branch_coverage=1 --extract coverage.info "*/boost/numeric/ublas/*" "*/libs/numeric/ublas/*" --output-file coverage.info
lcov --rc lcov_branch_coverage=1 --list coverage.info

