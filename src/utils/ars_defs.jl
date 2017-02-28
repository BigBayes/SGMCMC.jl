type HullNode
    m::Float64
    b::Float64
    left::Float64
    right::Float64
    pr::Float64

    #empty constructor
    function HullNode()
        return new()
    end
end

