
if !isdefined(:LOG_OUTPUT)
    global LOG_OUTPUT=STDOUT
end

function open_logfile(fn::String)
    global LOG_OUTPUT=open(fn,"w")
end

function close_logfile()
    close(LOG_OUTPUT)
end

function lprintln(s::String)
    write(LOG_OUTPUT, "$s\n")
end

function flush_log()
    flush(LOG_OUTPUT)
end
