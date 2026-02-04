# CPE 372 – High Performance Computing and Cloud Technologies

This repository contains programs and experiments related to **MPI (Message Passing Interface)** as part of the course **CPE 372: High Performance Computing and Cloud Technologies**.

The focus of this repository is to study and compare **blocking** and **non-blocking** communication in MPI, including performance behavior when running with different numbers of processes.

---

## Contents

- `block-mpi.c`  
  MPI program using **blocking communication** (e.g. `MPI_Send`, `MPI_Recv`).

- `non-block-mpi.c`  
  MPI program using **non-blocking communication** (e.g. `MPI_Isend`, `MPI_Irecv`, `MPI_Wait`).

---

## Requirements

- Linux (tested on Debian)
- MPI implementation:
  - OpenMPI **or**
  - MPICH
- GCC compiler

---

## Compilation

Use `mpicc` to compile the programs:

```bash
mpicc block-mpi.c -o block-mpi
mpicc non-block-mpi.c -o non-block-mpi
