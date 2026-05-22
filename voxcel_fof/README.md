# FoF halo finder

Classic friends-of-friends method for identifying groups of tracers using a grid (original version by Andres Ruiz).
Given the need to speed up execution to make this code competitive when used together with a CNN-based method vs a state-of-the-art halo finder, several optimizations have been implemented.

## Classic code (original)

Once the code is downloaded in the 'original' branch, we have the baseline version. The execution times are:


```
❯ ./main.x

 Lectura de trazadores
 | File = ../400.ascii
 | Numero de trazadores    = 769099 
 | Densidad media [h/Mpc]³ = 0.933401 
 | -> tiempo = 0.008634 min. 

 Armando linked-list
 | 1.0/NMEAN = 91.000 + 0.621 => NG = 91 
 | Lado del grid =  1.030 [Mpc/h] 
 | -> tiempo = 0.000129 min. 

 Comienza busqueda FoF
 | -> tiempo = 0.171881 min. 

 Escritura de grupos FoF
 | File = fof.groups
 | Total de grupos encontrados = 11653 
 | Total de grupos con mas de 10 miembros = 3149 

 | -> tiempo = 0.001033 min. 
```
Approximate wall time: 11 seconds (1 core 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz)

## Optimized grid (minecraft)

Based on voxel handling tools from the Popcorn Void Finder, a gridding method was implemented that explores grids without considering the fixed structure of central cell + 27 neighbors (the classic approach by Manuel in his FoF). This allows the grid to be as fine as desired and the neighbor search at the linking length scale can span several voxels. A function was implemented to tell whether a voxel is within the linking length, thus avoiding accessing unreachable voxels and improving execution time. In the 'minecraft' branch we have:

```
❯ ./main.x

 Lectura de trazadores
 | File = ../400.ascii
 | Numero de trazadores    = 769099 
 | Densidad media [h/Mpc]³ = 0.933401 
 | Read -> tiempo = 0.008419 min.

 Armando linked-list
 | NG = 256 
 | Lado del grid =  0.366 [Mpc/h] 
 | Grid -> tiempo = 0.000262 min.

 Comienza busqueda FoF
 | FoF -> tiempo = 0.049843 min.

 Escritura de grupos FoF
 | File = fof.groups
 | Total de grupos encontrados = 11653 
 | Total de grupos con mas de 10 miembros = 3149 

 | Write -> tiempo = 0.001048 min.

```
Wall time: 3.5 seconds (1 core 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz). Best NG = 256.

## Optimized grid + hash table (minecraft_hash)

Since very fine grids require more RAM, the grid header array of size ngrid^3 was replaced with a hash table that uses much less memory. However, this slightly increases execution time. In the 'minecraft_hash' branch:

```
❯ ./main.x

 Lectura de trazadores
 | File = ../400.ascii
 | Numero de trazadores    = 769099 
 | Densidad media [h/Mpc]³ = 0.933401 
 | read -> tiempo = 0.008578 min.

 Armando linked-list
 | NG = 512 
 | Lado del grid =  0.183 [Mpc/h] 
 | grid -> tiempo = 0.000709 min.

 Comienza busqueda FoF
 | FoF -> tiempo = 0.070105 min.

 Escritura de grupos FoF
 | File = fof.groups
 | Total de grupos encontrados = 11653 
 | Total de grupos con mas de 10 miembros = 3149 

 | write -> tiempo = 0.001074 min.

```
Wall time: approximately 4.8 seconds (1 core 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz)  
Only use when memory is the limiting factor. Best NG = 512.

## Optimized grid + Peano-Hilbert memory re-ordering (minecraft_peano)

Since using grids and linked lists results in memory access indirections, this leads to cache misses. The idea was to improve page faults by reordering according to a Peano-Hilbert space-filling curve over the grid, paying performance to reorder particles in memory for better access patterns during neighbor search. In the 'minecraft_peano' branch:

```
❯ ./main.x

 Lectura de trazadores
 | File = ../400.ascii
 | Numero de trazadores    = 769099 
 | Densidad media [h/Mpc]³ = 0.933401 
 | Read -> tiempo = 0.008851 min.
 | NG = 512 
 | Lado del grid =  0.183 [Mpc/h] 
 | Sort -> tiempo = 0.001634 min.

 Armando linked-list
 | Grid -> tiempo = 0.001726 min.

 Comienza busqueda FoF
 | FoF -> tiempo = 0.032751 min.

 Escritura de grupos FoF
 | File = fof.groups
 | Total de grupos encontrados = 11653 
 | Total de grupos con mas de 10 miembros = 3149 

 | Write -> tiempo = 0.000815 min.
```

Wall time: approximately 2.6 seconds (1 core 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz). Best NG = 512

## Optimized grid + Peano-Hilbert memory re-ordering + openmp (main)

Parallelizing the FoF algorithm is very challenging since it's a BFS-style tree exploration algorithm, which is inherently serial. To avoid decomposing space regions and starting from different seed tracers per core (which would require merging FoF lists through tracers in transition zones), the neighbor search at each BFS level was parallelized instead. That is, the descent to each level is done serially, and once a neighbor is dequeued to search its neighbors, this search is parallelized in independent queues via fork, then merged with a forced join. This introduces code serialization during synchronization stages and requires critical sections, which can also impact performance. In the 'main' branch:

```
❯ ./main.x

 Lectura de trazadores
 | File = ../400.ascii
 | Numero de trazadores    = 769099 
 | Densidad media [h/Mpc]³ = 0.933401 
 | Read -> tiempo = 0.008455 min.
 | NG = 512 
 | Lado del grid =  0.183 [Mpc/h] 
 | Sort -> tiempo = 0.001571 min.

 Armando linked-list
 | Grid -> tiempo = 0.001585 min.

 Comienza busqueda FoF
 | FoF -> tiempo = 0.009195 min.

 Escritura de grupos FoF
 | File = fof.groups
 | Total de grupos encontrados = 11653 
 | Total de grupos con mas de 10 miembros = 3149 

 | Write -> tiempo = 0.002735 min.

❯ ./main.x

 Lectura de trazadores
 | File = ../400.ascii
 | Numero de trazadores    = 769099 
 | Densidad media [h/Mpc]³ = 0.933401 
 | Read -> tiempo = 0.008751 min.
 | NG = 256 
 | Lado del grid =  0.366 [Mpc/h] 
 | Sort -> tiempo = 0.001358 min.

 Armando linked-list
 | Grid -> tiempo = 0.000251 min.

 Comienza busqueda FoF
 | FoF -> tiempo = 0.009508 min.

 Escritura de grupos FoF
 | File = fof.groups
 | Total de grupos encontrados = 11653 
 | Total de grupos con mas de 10 miembros = 3149 

 | Write -> tiempo = 0.001435 min.

```

Wall time: approximately 1.3 seconds (4 core 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz). Best NG = 256–512

## Future plan

The current code went from 11 seconds to 2.6 seconds in serial. With very simple parallelization, we gained another 2x using 4 cores, which leaves room for improving the parallelization.


