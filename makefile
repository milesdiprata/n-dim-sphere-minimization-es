# Compiler and linker
CC := gcc

# The target binary program
TARGET := a.out

# The directories, source, includes, objects, binary and resources
SRCDIR := src
INCDIR := inc
BUILDDIR := obj
TARGETDIR := bin
RESDIR := res
SRCEXT := cc
DEPEXT := d
OBJEXT := o

# Flags, libraries and includes
CFLAGS := -Wall -Werror -std=c++20 -g
LIB :=
INC := -I$(SRCDIR) -I$(INCDIR) -I/usr/local/include
INCDEP := -I$(SRCDIR) -I$(INCDIR)

# Sources and objects
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.$(OBJEXT)))

# Default make
all: resources $(TARGET)

# Remake
remake: cleaner all

# Copy resources from resources directory to target directory
resources: directories
	@cp -r $(RESDIR)/. $(TARGETDIR)/

# Make the directories
directories:
	@mkdir -p $(TARGETDIR)
	@mkdir -p $(BUILDDIR)
	@mkdir -p $(RESDIR)

# Clean only objects
clean:
	@$(RM) -rf $(BUILDDIR)

# Full clean, objects and binaries
cleaner: clean
	@$(RM) -rf $(TARGETDIR)

# Run the default target
run: all
	./$(TARGETDIR)/$(TARGET)

# Pull in dependency info for *existing* .o files
-include $(OBJECTS:.$(OBJEXT)=.$(DEPEXT))

# Link
$(TARGET): $(OBJECTS)
	$(CXX) -o $(TARGETDIR)/$(TARGET) $^ $(LIB)

# Compile
$(BUILDDIR)/%.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<
	@$(CC) $(CFLAGS) $(INCDEP) -MM $(SRCDIR)/$*.$(SRCEXT) > $(BUILDDIR)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR)/$*.$(DEPEXT) $(BUILDDIR)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR)/$*.$(OBJEXT):|' < $(BUILDDIR)/$*.$(DEPEXT).tmp > $(BUILDDIR)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILDDIR)/$*.$(DEPEXT).tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILDDIR)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR)/$*.$(DEPEXT).tmp

# Non-file targets
.PHONY: all remake clean cleaner resources run
