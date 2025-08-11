#!/bin/bash

# Functions available in this file include:
# - setup_obspack

# Description: Build ObsPack files
# Usage:
#   setup_obspack
setup_obspack() {
    printf "\n=== SETTING UP ObsPack DIRECTORY ===\n"

    cd ${OutputPath}/$RunName
    mkdir -p -v obspack_data

    python ${InversionPath}/src/components/obspack_component/build_obspack_files.py $DataPathObsPack $LatMinInvDomain $LatMaxInvDomain $LonMinInvDomain $LonMaxInvDomain $StartDate $EndDate

    printf "\n=== DONE SETTING UP ObsPack DIRECTORY ===\n"
}
