
# Define Fortran compiler
FC = pgfortran

all: assignment_3.out

assignment_3.out: assignment_3.cuf helper.o
	$(FC) -Mcuda=cc3x -o assignment_3.out assignment_3.cuf helper.o

helper.o: helper.cu
	nvcc -c helper.cu

clean:
	@$(RM) *.out *.o *.a *~ *.tmp *.mod

clean_all: clean
	@$(RM) -rf images/*result*.bmp images/*result*.jpg

rebuild: clean all


