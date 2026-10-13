# Build AND execute a notebook from a percent-format .jl file, embedding outputs.
#
#   julia sweeps/tools/run_nb.jl <cells.jl> <out.ipynb>            (build + execute)
#   julia sweeps/tools/run_nb.jl <cells.jl> <out.ipynb> --no-run   (build only, no outputs)
#
# Cell markers in the .jl file:
#   # %% [markdown]      following lines are markdown, each prefixed with "# "
#   # %%                 following lines are a Julia code cell
# Code cells run in order in one module, with @__DIR__ = folder of out.ipynb.
# Captured: printed output and log messages, and the cell's final value
# (Makie figures as PNG, tables as HTML + text). A trailing `;` suppresses the value.
# No Jupyter needed; no extra packages (JSON is written by hand).

using Base64, Logging

# --- parse percent-format cells ----------------------------------------------------
function parse_cells(path)
    cells = Pair{String, Vector{String}}[]
    for line in eachline(path)
        if startswith(line, "# %% [markdown]")
            push!(cells, "markdown" => String[])
        elseif startswith(line, "# %%")
            push!(cells, "code" => String[])
        elseif !isempty(cells)
            kind, lines = cells[end]
            push!(lines, kind == "markdown" ? replace(line, r"^# ?" => "") : line)
        end
    end
    trim(ls) = (while !isempty(ls) && isempty(strip(ls[end])); pop!(ls); end;
                while !isempty(ls) && isempty(strip(ls[1])); popfirst!(ls); end; ls)
    return [k => join(trim(ls), "\n") for (k, ls) in cells]
end

# --- minimal JSON writer -----------------------------------------------------------
function jstr(s::AbstractString)
    io = IOBuffer(); print(io, '"')
    for c in s
        if c == '"'; print(io, "\\\"")
        elseif c == '\\'; print(io, "\\\\")
        elseif c == '\n'; print(io, "\\n")
        elseif c == '\r'; print(io, "\\r")
        elseif c == '\t'; print(io, "\\t")
        elseif c < ' '; print(io, "\\u", string(UInt16(c); base = 16, pad = 4))
        else print(io, c)
        end
    end
    print(io, '"'); return String(take!(io))
end
json(x::AbstractString) = jstr(x)
json(x::Nothing) = "null"
json(x::Bool) = x ? "true" : "false"
json(x::Integer) = string(x)
json(x::AbstractVector) = "[" * join(json.(x), ",") * "]"
json(x::AbstractDict) = "{" * join(["$(jstr(string(k))):$(json(v))" for (k, v) in x], ",") * "}"
json(x::Pair...) = json(Dict(x))
"Notebook-style multiline string: list of lines keeping their newlines."
srclines(s) = (ls = split(s, '\n'); [i < length(ls) ? l * "\n" : l for (i, l) in enumerate(ls)])

strip_ansi(s) = replace(s, r"\e\[[0-9;]*[A-Za-z]" => "")

# --- run ----------------------------------------------------------------------------
function run_notebook(cells_path, out_path; execute = true)
    out_path = abspath(out_path)
    cells = parse_cells(cells_path)
    NB = Module(:Notebook)
    Core.eval(NB, :(include(p) = Base.include($NB, p)))   # bare modules lack include/eval
    Core.eval(NB, :(eval(x) = Core.eval($NB, x)))
    cell_json = String[]
    count = 0
    failed = false
    if !execute     # just build the notebook: code cells empty, ready to run in VS Code
        for (i, (kind, src)) in enumerate(cells)
            id = "\"id\":" * jstr("c" * lpad(i, 3, '0'))
            push!(cell_json, kind == "markdown" ?
                  "{\"cell_type\":\"markdown\",$id,\"metadata\":{},\"source\":$(json(srclines(src)))}" :
                  "{\"cell_type\":\"code\",\"execution_count\":null,$id,\"metadata\":{},\"outputs\":[],\"source\":$(json(srclines(src)))}")
        end
        failed = true   # skips the execution loop below
        cells = Pair{String, String}[]
    end
    for (i, (kind, src)) in enumerate(cells)
        id = "\"id\":" * jstr("c" * lpad(i, 3, '0'))
        if kind == "markdown"
            push!(cell_json, "{\"cell_type\":\"markdown\",$id,\"metadata\":{},\"source\":$(json(srclines(src)))}")
            continue
        end
        outputs = String[]
        if !failed
            count += 1
            t0 = time()
            logfile = tempname()
            val, err = nothing, nothing
            open(logfile, "w") do io
                redirect_stdout(io) do
                    redirect_stderr(io) do
                        with_logger(ConsoleLogger(io)) do
                            try
                                val = Base.include_string(NB, src, out_path)
                            catch e
                                err = (e, catch_backtrace())
                            end
                        end
                    end
                end
            end
            text = strip_ansi(read(logfile, String)); rm(logfile)
            isempty(text) || push!(outputs, "{\"output_type\":\"stream\",\"name\":\"stdout\",\"text\":$(json(srclines(text)))}")
            if err !== nothing
                msg = strip_ansi(sprint(showerror, err[1], err[2]))
                push!(outputs, "{\"output_type\":\"error\",\"ename\":$(jstr(string(typeof(err[1])))),\"evalue\":$(jstr(first(msg, 2000))),\"traceback\":$(json(srclines(first(msg, 6000))))}")
                @warn "cell $i failed; later cells left unexecuted" exception = err[1]
                failed = true
            elseif val !== nothing && !endswith(rstrip(src), ";")
                data = Dict{String, Any}()
                if Base.invokelatest(showable, MIME"image/png"(), val) && !(val isa AbstractString)
                    f = tempname() * ".png"
                    Core.eval(NB, :(CairoMakie.save($f, $val; px_per_unit = 2)))   # eval: latest world
                    data["image/png"] = base64encode(read(f)); rm(f)
                    data["text/plain"] = [string(typeof(val))]
                else
                    ctx(io) = IOContext(io, :limit => true, :displaysize => (60, 200), :color => false)
                    data["text/plain"] = srclines(Base.invokelatest(sprint, (io, v) -> show(ctx(io), MIME"text/plain"(), v), val))
                    if !(val isa AbstractString) && Base.invokelatest(showable, MIME"text/html"(), val)
                        data["text/html"] = srclines(Base.invokelatest(sprint, (io, v) -> show(ctx(io), MIME"text/html"(), v), val))
                    end
                end
                push!(outputs, "{\"output_type\":\"execute_result\",\"execution_count\":$count,\"data\":$(json(data)),\"metadata\":{}}")
            end
            println(stdout, "cell $i done ($(round(time() - t0; digits = 1)) s)"); flush(stdout)
        end
        ec = isempty(outputs) && failed ? "null" : string(count)
        push!(cell_json, "{\"cell_type\":\"code\",\"execution_count\":$ec,$id,\"metadata\":{},\"outputs\":[$(join(outputs, ","))],\"source\":$(json(srclines(src)))}")
    end
    meta = """{"kernelspec":{"display_name":"Julia 1.12.6","language":"julia","name":"julia-1.12"},"language_info":{"file_extension":".jl","mimetype":"application/julia","name":"julia","version":"1.12.6"}}"""
    write(out_path, "{\"cells\":[\n" * join(cell_json, ",\n") * "\n],\"metadata\":$meta,\"nbformat\":4,\"nbformat_minor\":5}\n")
    println("wrote $out_path  ($(length(cell_json)) cells, executed = $execute, failed = $(execute && failed))")
    return !execute || !failed
end

if abspath(PROGRAM_FILE) == @__FILE__
    ok = run_notebook(ARGS[1], ARGS[2]; execute = !("--no-run" in ARGS))
    exit(ok ? 0 : 1)
end
