import multiprocessing
import numpy as np
from mpi4py import MPI


def mpi_split(size, comm):
    '''Local sizes of size-d object on processes of comm'''
    ncpus = comm.size
    sizes = []
    for rank in range(ncpus):
        sizes.append((size + ncpus - rank - 1)/(ncpus - rank))
        size -= sizes[-1]
    return sizes


def distribute(words, comm):
    '''Something like Scatterv of words accross processes in comm'''
    world_size = comm.size
    # The idea is to let each process receive a chunk of characters and
    # map which for each word in the chunk has its size - this is enough
    # to reconstruct the word
    local_nwords = np.empty(world_size, dtype=int)
    # Root compresses and scatter
    if comm.rank == 0:
        nwords = len(words)
        # Partition words simply by their number
        local_nwords[:] = mpi_split(nwords, comm)
        # Let each process to know how many words to expect. Will receive
        # int for each representing the word size
        comm.Bcast([local_nwords, MPI.LONG], root=0)

        # The root now prepares the chunking
        # First how
        word_lens = np.array(map(len, words), dtype=int)
        word_offsets = np.cumsum(np.r_[0, local_nwords[:-1]])
        
        local_word_lens = np.zeros(local_nwords[comm.rank], dtype=int)
        comm.Scatterv([word_lens, local_nwords, word_offsets, MPI.LONG],
                      local_word_lens, root=0)

        # Now the word_chunks
        word_offsets = np.r_[word_offsets, word_offsets[-1] + local_nwords[-1]]

        words = [''.join(words[f:l])
                 for f, l in zip(word_offsets[:-1], word_offsets[1:])]
        my_words = comm.scatter(words, root=0)
    # Just wait
    else:
        # How many words
        comm.Bcast([local_nwords, MPI.LONG], root=0)
        # Get bounds for decoding
        # FIXME: isn't it faster to send the actual individuals around and
        # not compile?
        local_word_lens = np.zeros(local_nwords[comm.rank], dtype=int)
        comm.Scatterv([None, None, None, MPI.LONG], local_word_lens, root=0) 
        # Get my words
        my_words = comm.scatter(None, root=0)

    # Now we can decode
    bounds = np.cumsum(np.r_[0, local_word_lens])
    my_words = [my_words[f:l] for f, l in zip(bounds[:-1], bounds[1:])]

    return my_words, local_nwords


def collect(my_values, comm, sizes=None):
    '''Gatherv values from comm's processes to root'''
    assert my_values.dtype == float
    
    # Don't know about sizes on other procs?
    if sizes is None:
        sizes = np.empty(comm.size, dtype=int)
        size = np.array([len(my_values)], dtype=int)
        comm.Allgather(size, sizes)
    
    offsets = np.cumsum(np.r_[0, sizes[:-1]])
    # Let root alloc
    if comm.rank == 0:
        values = np.empty(sum(sizes), dtype=float)
    else:
        values = None

    comm.Gatherv([my_values, sizes[comm.rank], MPI.DOUBLE],
                 [values, sizes, offsets, MPI.DOUBLE], root=0)

    return values


def fitness(word): return sum(map(ord, word))

# --------------------------------------------------------------------

if __name__ == '__main__':

    nthreads = 4
    # Operation to be performed on word
    pool = multiprocessing.Pool(processes=nthreads)
    
    comm = MPI.COMM_WORLD
    # Let root make the words
    if comm.rank == 0:
        words = []
        with open('words.txt') as f:
            for line in f:
                words.append(line.strip())

        words = words
        for i in range(4):
            words.extend(words)
        print 'Words is ready', len(words)
    else:
        words = None
                
    my_words, local_nwords = distribute(words, comm)

    # Some local computations
    my_values = np.array(pool.map(fitness, my_words), dtype=float)

    # Gather
    values = collect(my_values, comm, sizes=local_nwords)
    # Finish
    if comm.rank == 0: print sum(values)

    pool.close()
