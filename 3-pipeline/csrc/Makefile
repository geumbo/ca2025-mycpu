CROSS_COMPILE ?= $(HOME)/riscv-none-elf-gcc/bin/riscv-none-elf-

ASFLAGS = -march=rv32i_zicsr -mabi=ilp32
CFLAGS = -O0 -Wall -march=rv32i_zicsr -mabi=ilp32
LDFLAGS = --oformat=elf32-littleriscv

AS := $(CROSS_COMPILE)as
CC := $(CROSS_COMPILE)gcc
LD := $(CROSS_COMPILE)ld
OBJCOPY := $(CROSS_COMPILE)objcopy

%.o: %.S
	$(AS) -R $(ASFLAGS) -o $@ $<
%.elf: %.S
	$(AS) -R $(ASFLAGS) -o $(@:.elf=.o) $<
	$(CROSS_COMPILE)ld -o $@ -T link.lds $(LDFLAGS) $(@:.elf=.o)
%.elf: %.c init.o
	$(CC) $(CFLAGS) -c -o $(@:.elf=.o) $<
	$(CROSS_COMPILE)ld -o $@ -T link.lds $(LDFLAGS) $(@:.elf=.o) init.o

# Special rule for uf8_test: link with uf8.o assembly
uf8_test.elf: uf8_test.c uf8.o init.o
	$(CC) $(CFLAGS) -c -o uf8_test.o uf8_test.c
	$(CROSS_COMPILE)ld -o $@ -T link.lds $(LDFLAGS) uf8_test.o uf8.o init.o

%.asmbin: %.elf
	$(OBJCOPY) -O binary -j .text -j .data $< $@

BINS = \
	fibonacci.asmbin \
	hazard.asmbin \
	quicksort.asmbin \
	sb.asmbin \
	uart.asmbin \
	irqtrap.asmbin \
	uf8_test.asmbin

# Clear the .DEFAULT_GOAL special variable, so that the following turns
# to the first target after .DEFAULT_GOAL is not set.
.DEFAULT_GOAL :=

all: $(BINS)

update: $(BINS)
	cp -f $(BINS) ../src/main/resources

clean:
	$(RM) *.o *.elf *.asmbin
