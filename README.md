# FDD3360
Course on Applied GPU programming at KTH

This repository contains the solution/codes for the assignments in the course.

The codes are written in Fortran. cudaFortran is supported by PGI.
The codes are compiled as follows:

**module load pgi cuda**
to load the PGI compiler.

To compile the cudaFortran code:
**pgfortran -Mcuda=cc3x cuda_file.cuf -o cuda_file.out**


Update: Only Assignment 3 - exercise 3 and bonus exercise features cudaC code.
This is because of the asynchronous copy of derived data type where, in cudaFortran the datatype has to be generic.
As a temporary fix, switched to cudaC in only those exercises. A better solution should be available and shall be updated, when possible.
