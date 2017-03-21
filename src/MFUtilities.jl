module MFUtilities

    export comp_inv_pr_pick, get_n_elem, BatchIterator

    function get_n_elem(block, nusers, nitems)
        n_elem_row = zeros(Int64,nusers)
        n_elem_col = zeros(Int64,nitems)
        n = size(block)[1]
        for j in 1:n
            u,i,r = block[j,:]
            u = round(Int64,u)
            i = round(Int64,i)
            n_elem_row[u] += 1
            n_elem_col[i] += 1
        end
        return n_elem_row, n_elem_col
    end


    # compute local inverse pick prob
    function comp_inv_pr_pick(n_elem_row,n_elem_col,batchsize)
        N = sum(n_elem_row)
        try
            @assert N == sum(n_elem_col)
        catch
            println("$(myid()) assertion failed")
        end

        Pr_us = 1 - (1 - n_elem_row./ N ) .^ batchsize
        Pr_is = 1 - (1 - n_elem_col./ N ) .^ batchsize

        return 1 ./ Pr_us, 1 ./ Pr_is
    end

    """
    BatchIterator type to get minibatches
    """
    type BatchIterator
      nitems::Int
      batchsize::Int
      niters::Float64 # float to allow infinite iterations.
      randomize::Bool

      function BatchIterator(nitems::Int,batchsize::Int,niters::Float64;randomize::Bool=false)
        @assert batchsize > 0
        @assert nitems >= batchsize
        @assert niters >= 0
        new(nitems,batchsize,niters,randomize)
      end
    end

    function Base.start(B::BatchIterator)
      (0,0,B.randomize ? randperm(B.nitems) : (1:B.nitems))
    end
    function Base.next(B::BatchIterator,state)
      iter = state[1]
      id = state[2]
      order = state[3]

      # +1 is necessary since the Base.start uses a 0 index
      # The remainder means that the batches wrap around if the batch size doesn't evenly divide the data size
      batch = order[rem(Int[i for i = id:id+B.batchsize-1],B.nitems)+1]

      # get new permutation for next epoch
      if id+B.batchsize >= B.nitems
        if B.randomize
          order = randperm(B.nitems)
        end
      end

      id = rem(id+B.batchsize,B.nitems)
      iter += 1

      # Return the indices of the next minibatch, and update the state (which actually points to the next minibatch)
      (batch,(iter,id,order))
    end

    # Iterating is done when the iteration in the state matches the preset iteration limit in the iterator object
    function Base.done(B::BatchIterator,state)
      state[1] >= B.niters
    end


end
