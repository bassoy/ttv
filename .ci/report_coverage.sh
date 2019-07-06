#! /bin/bash -e

# Only collect and upload coverage reports from C++17 Builds
# This is required because Tensor Extension only runs with C++17

# Capture all the coverge reports
#lcov --rc lcov_branch_coverage=1 --directory ${TLIB_ROOT} --capture --output-file coverage.info 
lcov -c -i --rc lcov_branch_coverage=1 -d . -o Coverage.baseline


./bin/main --gtest_filter="-TensorTimesVector.CheckIndexDivisionSmallBlock"

#--no-external 
    
# Remove all unwanted coverages libs. 
# Boost.uBLAS depends uses many internal boost libs, we don't want them to be in coverage.


lcov -c --rc lcov_branch_coverage=1 -d . -o Coverage.out #-b .. 
lcov -a Coverage.baseline -a Coverage.out -o Coverage.combined

lcov --extract Coverage.combined "*/ttv/*" --output-file Coverage.combined


genhtml Coverage.combined -o HTML
chromium-browser HTML/index.html

#lcov --rc lcov_branch_coverage=1 --extract coverage.info "*/boost/numeric/ublas/*" "*/libs/numeric/ublas/*" --output-file coverage.info
#lcov --rc lcov_branch_coverage=1 --list coverage.info

