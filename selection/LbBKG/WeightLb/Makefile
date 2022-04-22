#
# Makefile for programs that include ROOT and RooFit.
#
ROOTCFLAGS   := $(shell ${ROOTSYS}/bin/root-config --cflags)
ROOTLIBS     := $(shell ${ROOTSYS}/bin/root-config --libs)

CXXFLAGS      = -O3 -Wall -fPIC -g -W 

CXXFLAGS     += -I. $(ROOTCFLAGS)

LIBS       += $(ROOTLIBS) -lHtml -lThread -lMinuit -lRooFit -lRooFitCore -lMathCore -lMinuit2 -lRooStats -lMathMore
# -lEve -lGeom -lGui

SRCDIR        = src/

BINDIR        = bin/

SCRDIR	      = scripts/

Header        = RooTDalitz/

CC            = g++

OBJS	      = $(BINDIR)$(foo).o $(BINDIR)RooDalitzAmplitude.o $(BINDIR)EvtCGCoefSingle.o $(BINDIR)EvtOrthogVector.o $(BINDIR)RooSUMVar.o


#-------------------------------------------------------
all:				$(foo)
#-------------------------------------------------------

$(foo) :			$(OBJS) 
				@echo "Linking executable ..."
				${CC} $(LIBS) -o $(BINDIR)$(foo) $(OBJS) 
				@echo "Done."
				#cp $(BINDIR)$(foo) ../bin/$(foo)
                                   
#-------------------------------------------------------
$(BINDIR)$(foo).o : 		${SCRDIR}$(foo).C
				#@mkdir -p bin
				@echo Compiling components...
				${CC} $(CXXFLAGS) -c ${SCRDIR}$(foo).C -o $(BINDIR)$(foo).o


$(BINDIR)RooDalitzAmplitude.o:	${SRCDIR}RooDalitzAmplitude.cxx ${Header}RooDalitzAmplitude.h
				${CC} $(CXXFLAGS) -c ${SRCDIR}RooDalitzAmplitude.cxx -o $(BINDIR)RooDalitzAmplitude.o

$(BINDIR)EvtCGCoefSingle.o:	${SRCDIR}EvtCGCoefSingle.cpp ${Header}EvtCGCoefSingle.hh
				${CC} $(CXXFLAGS) -c ${SRCDIR}EvtCGCoefSingle.cpp -o $(BINDIR)EvtCGCoefSingle.o

$(BINDIR)EvtOrthogVector.o:	${SRCDIR}EvtOrthogVector.cpp ${Header}EvtOrthogVector.hh
				${CC} $(CXXFLAGS) -c ${SRCDIR}EvtOrthogVector.cpp -o $(BINDIR)EvtOrthogVector.o

$(BINDIR)RooSUMVar.o:		${SRCDIR}RooSUMVar.cxx ${Header}RooSUMVar.h
				${CC} $(CXXFLAGS) -c ${SRCDIR}RooSUMVar.cxx -o $(BINDIR)RooSUMVar.o


#$(BINDIR)RooIpatia2.o : 	${SRCDIR}RooIpatia2.cxx ${Header}RooIpatia2.h
#				${CC} $(CXXFLAGS) -c ${SRCDIR}RooIpatia2.cxx -o $(BINDIR)RooIpatia2.o

#-------------------------------------------------------
clean:
		@rm -f bin/tpanal ${SRCDIR}._* ${SRCDIR}*~ core* $(BINDIR)/*.o $(BINDIR)/*.so ._*
		@echo cleaned
# DO NOT DELETE
