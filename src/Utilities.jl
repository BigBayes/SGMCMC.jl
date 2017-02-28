module Utilities


export @dict

# Say there are variables a,b,c in scope, with values x,y,z.
# @dict(a,b,c) returns [:a=>x,:b=>y,:c=>z]
macro dict(args...)
  Expr(:call,:Dict, [Expr(:(=>),Expr(:quote,arg),esc(arg)) for arg in args]...)
end

function set(specs::Dict; keyargs...)
  for (k,v) in keyargs
    specs[k] = v
  end
  specs
end

end
