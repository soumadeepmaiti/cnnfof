#__________________________________________________________________
#                                                          Compiler
                                                  
#CC = g++ -fopenmp -fsanitize=thread -g -O1
CC = g++ -O3 -fopenmp 


#__________________________________________________________________
#

EXEC = main.x

OBJS = allvars.o t.o io.o grid.o fof.o main.o

INCL = allvars.h proto.h Makefile

LIBS = -lm

$(EXEC): $(OBJS) 
	$(CC) $(OBJS) $(LIBS) -o $(EXEC)  

$(OBJS): $(INCL) 

clean:
	rm -f $(OBJS) $(EXEC) *~ 
