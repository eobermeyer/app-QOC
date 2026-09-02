########################################################################
# Application web interactive pour l'optimisation GRAPE d'un système    #
# quantique à 2 niveaux (basée sur le script QOC_fun.jl fourni).        #
#                                                                        #
# Framework : Genie.jl (le plus simple pour un formulaire + calcul +    #
# affichage de résultats, sans dépendre du format de notebook Pluto).   #
########################################################################

using Genie
using Genie.Router
using Genie.Renderer.Html
using Genie.Requests
using Base64: base64encode

using GRAPE
using QuantumPropagators
using QuantumPropagators: hamiltonian, ExpProp, propagate
using QuantumControl.Functionals: J_T_sm, J_a_fluence
using LinearAlgebra
using Plots

# Backend GR sans affichage (obligatoire dans un conteneur headless)
ENV["GKSwstype"] = "100"
gr()

# ------------------------------------------------------------------
# Éléments physiques repris du script original
# ------------------------------------------------------------------

const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]

operator_from_name(name::AbstractString) =
    name == "sigma_x" ? σx :
    name == "sigma_y" ? σy :
    name == "sigma_z" ? σz :
    error("Opérateur inconnu : $name")

state_from_choice(choice::AbstractString) =
    choice == "0" ? ComplexF64[1, 0] :
    choice == "1" ? ComplexF64[0, 1] :
    begin
        v = rand(ComplexF64, 2)
        v ./ norm(v)
    end

FS_angle(phi, psi) = acos(clamp(abs(dot(phi, psi)), 0.0, 1.0))

function FS_geod(t::Real, phi, psi)
    θ = angle(dot(phi, psi))
    psi2 = exp(-im * θ) * psi
    α = FS_angle(phi, psi2)
    α < 1e-12 && return phi
    sin((1 - t) * α) / sin(α) * phi + sin(t * α) / sin(α) * psi2
end

pop(ψ, states) = [abs2(dot(ψ, ϕ)) for ϕ in eachcol(states)]

# Accès "sûr" à un champ qui peut ne pas exister selon la version des
# paquets JuliaQuantumControl installée (on ne fait jamais planter la
# page de résultats pour un champ optionnel).
safe_field(obj, sym::Symbol, default) = hasproperty(obj, sym) ? getfield(obj, sym) : default

# ------------------------------------------------------------------
# Cœur du calcul : reprend fidèlement QOC_fun.jl, mais paramétré
# ------------------------------------------------------------------

function run_optimization(p::Dict)
    ψ0 = state_from_choice(p[:psi0])
    ψ1 = state_from_choice(p[:psi1])

    op1 = operator_from_name(p[:op1])
    op2 = operator_from_name(p[:op2])
    H_mat = [p[:coeff1] .* op1, p[:coeff2] .* op2]

    u1_0 = p[:u1_init]
    u2_0 = p[:u2_init]
    u1(t) = u1_0
    u2(t) = u2_0

    H_comp = [(H_mat[1], u1), (H_mat[2], u2)]
    H = hamiltonian(H_comp...)

    tlist = collect(range(0.0, p[:t_final]; length = p[:n_points]))

    traj = GRAPE.Trajectory(
        initial_state = ψ0,
        generator = H,
        target_state = ψ1,
    )

    check_convergence = res -> (res.J_T < 1e-5 ? "J_T < 1e-5 atteint" : false)

    result = GRAPE.optimize(
        [traj],
        tlist;
        prop_method = ExpProp,
        J_T = J_T_sm,
        J_a = J_a_fluence,
        lambda_a = p[:lambda_a],
        iter_stop = p[:max_iter],
        check_convergence = check_convergence,
    )

    u_opt = result.optimized_controls
    H_comp_opt = [(H_mat[1], u_opt[1]), (H_mat[2], u_opt[2])]
    H_opt = hamiltonian(H_comp_opt...)

    states = propagate(ψ0, H_opt, tlist; method = ExpProp, storage = true)

    ϕ1 = states[:, end]
    c = dot(ψ1, ϕ1)
    fidelity = abs2(c)
    α = angle(c)
    err = norm(ϕ1 - exp(im * α) * ψ1)

    fs_dist = FS_angle(ψ0, ψ1)
    fs_diff = abs(fs_dist - sqrt(max(result.J_a, 0.0)))

    # --- Figure 1 : évolution des populations avec contrôles optimisés
    P0 = pop(ψ0, states)
    P1 = pop(ψ1, states)
    plt1 = plot(
        tlist, P0;
        xlabel = "t", ylabel = "Population",
        label = "|⟨ψ0|φ(t)⟩|²", title = "Évolution avec contrôles optimisés",
        lw = 2, legend = :best,
    )
    plot!(plt1, tlist, P1; label = "|⟨ψ1|φ(t)⟩|²", lw = 2)

    # --- Figure 2 : comparaison avec la géodésique de Fubini-Study
    states_FS = stack(FS_geod.(tlist, Ref(ψ0), Ref(ψ1)))
    P0g = pop(ψ0, states_FS)
    P1g = pop(ψ1, states_FS)
    plt2 = plot(
        tlist, P0g;
        xlabel = "t", ylabel = "Population",
        label = "|⟨ψ0|φ(t)⟩|²", title = "Géodésique de Fubini-Study",
        lw = 2, legend = :best,
    )
    plot!(plt2, tlist, P0g; label = false, lw = 0) # no-op garde-fou
    plot!(plt2, tlist, P1g; label = "|⟨ψ1|φ(t)⟩|²", lw = 2)

    # --- Figure 3 : contrôles optimisés u1(t), u2(t)
    t_mid = (tlist[1:end-1] .+ tlist[2:end]) ./ 2
    u1v = u_opt[1]
    u2v = u_opt[2]
    tax1 = length(u1v) == length(t_mid) ? t_mid : tlist[1:length(u1v)]
    tax2 = length(u2v) == length(t_mid) ? t_mid : tlist[1:length(u2v)]
    plt3 = plot(
        tax1, real.(u1v);
        xlabel = "t", ylabel = "Amplitude",
        label = "u1(t)", title = "Contrôles optimisés (partie réelle)",
        lw = 2, seriestype = :steppost, legend = :best,
    )
    plot!(plt3, tax2, real.(u2v); label = "u2(t)", lw = 2, seriestype = :steppost)

    return (
        J_T = result.J_T,
        J_a = result.J_a,
        fidelity = fidelity,
        error = err,
        fs_diff = fs_diff,
        n_iter = safe_field(result, :iter, nothing),
        message = safe_field(result, :message, nothing),
        plots = (plt1, plt2, plt3),
    )
end

function plot_to_data_uri(plt)
    io = IOBuffer()
    show(io, MIME("image/png"), plt)
    "data:image/png;base64," * base64encode(take!(io))
end

# ------------------------------------------------------------------
# Aide pour lire et valider les paramètres du formulaire
# ------------------------------------------------------------------

function parse_params()
    getf(name, default) = begin
        raw = postpayload(Symbol(name), string(default))
        try
            parse(Float64, raw)
        catch
            default
        end
    end
    geti(name, default) = Int(round(getf(name, default)))

    Dict(
        :psi0 => postpayload(:psi0, "0"),
        :psi1 => postpayload(:psi1, "1"),
        :op1 => postpayload(:op1, "sigma_x"),
        :op2 => postpayload(:op2, "sigma_y"),
        :coeff1 => getf("coeff1", -1.0),
        :coeff2 => getf("coeff2", 1000.0),
        :u1_init => getf("u1_init", 0.1),
        :u2_init => getf("u2_init", 0.1),
        :t_final => clamp(getf("t_final", 1.0), 1e-3, 1000.0),
        :n_points => clamp(geti("n_points", 101), 3, 2001),
        :lambda_a => clamp(getf("lambda_a", 1e-5), 0.0, 1e6),
        :max_iter => clamp(geti("max_iter", 300), 1, 3000),
    )
end

# ------------------------------------------------------------------
# Vues HTML (tout dans un seul fichier pour rester simple)
# ------------------------------------------------------------------

const PAGE_STYLE = """
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0f1220; color:#e8e8f0; margin:0; padding:0 0 60px 0; }
  .wrap { max-width: 880px; margin: 0 auto; padding: 32px 20px; }
  h1 { font-size: 1.6rem; margin-bottom: 4px; }
  p.sub { color:#a0a3c0; margin-top:0; }
  fieldset { border: 1px solid #333653; border-radius: 10px; margin-bottom: 18px; padding: 16px 20px; }
  legend { padding: 0 8px; color:#9fd4ff; font-weight:600; }
  label { display:block; font-size:0.85rem; color:#c3c6e6; margin-top:10px; margin-bottom:4px; }
  input, select { width:100%; box-sizing:border-box; padding:8px 10px; border-radius:6px; border:1px solid #3a3e63; background:#1a1d34; color:#eee; font-size:0.95rem; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:0 20px; }
  button { background:#5468ff; color:white; border:none; padding:12px 22px; border-radius:8px; font-size:1rem; cursor:pointer; margin-top:10px; }
  button:hover { background:#6c7cff; }
  a.btn { display:inline-block; background:#333653; color:#e8e8f0; padding:10px 18px; border-radius:8px; text-decoration:none; margin-top:20px; }
  table.results { width:100%; border-collapse: collapse; margin: 16px 0 28px 0; }
  table.results td { padding:8px 10px; border-bottom:1px solid #2a2d4a; }
  table.results td:first-child { color:#9fd4ff; width:60%; }
  .figure { background:#161a30; border-radius:10px; padding:14px; margin-bottom:20px; text-align:center; }
  .figure img { max-width:100%; border-radius:6px; }
  .err { background:#3a1a24; border:1px solid #79263c; color:#ffb3c0; padding:14px 18px; border-radius:10px; }
</style>
"""

function form_html(p::Dict = Dict())
    g(k, d) = get(p, k, d)
    """
    <!doctype html><html lang="fr"><head><meta charset="utf-8">
    <title>Contrôle optimal quantique — GRAPE</title>$PAGE_STYLE</head>
    <body><div class="wrap">
      <h1>Contrôle optimal quantique (GRAPE)</h1>
      <p class="sub">Système à 2 niveaux, H(t) = c1·op1 + c2·op2 pondérés par ε(t). Ajustez les paramètres puis lancez l'optimisation.</p>
      <form action="/run" method="post">
        <fieldset>
          <legend>États</legend>
          <div class="grid">
            <div>
              <label>État initial ψ0</label>
              <select name="psi0">
                <option value="0">|0⟩</option>
                <option value="1">|1⟩</option>
                <option value="random">Aléatoire</option>
              </select>
            </div>
            <div>
              <label>État cible ψ1</label>
              <select name="psi1">
                <option value="0">|0⟩</option>
                <option value="1" selected>|1⟩</option>
                <option value="random">Aléatoire</option>
              </select>
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>Hamiltonien H(t) = c1·op1 + c2·op2 (pondéré par les contrôles)</legend>
          <div class="grid">
            <div>
              <label>Opérateur 1</label>
              <select name="op1">
                <option value="sigma_x" selected>σx</option>
                <option value="sigma_y">σy</option>
                <option value="sigma_z">σz</option>
              </select>
              <label>Coefficient c1</label>
              <input type="number" step="any" name="coeff1" value="-1">
              <label>ε1 initial (constant)</label>
              <input type="number" step="any" name="u1_init" value="0.1">
            </div>
            <div>
              <label>Opérateur 2</label>
              <select name="op2">
                <option value="sigma_x">σx</option>
                <option value="sigma_y" selected>σy</option>
                <option value="sigma_z">σz</option>
              </select>
              <label>Coefficient c2</label>
              <input type="number" step="any" name="coeff2" value="1000">
              <label>ε2 initial (constant)</label>
              <input type="number" step="any" name="u2_init" value="0.1">
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>Grille temporelle &amp; optimisation</legend>
          <div class="grid">
            <div>
              <label>Temps final T</label>
              <input type="number" step="any" name="t_final" value="1.0">
              <label>Nombre de points</label>
              <input type="number" step="1" name="n_points" value="101">
            </div>
            <div>
              <label>λa (pénalité fluence)</label>
              <input type="number" step="any" name="lambda_a" value="0.00001">
              <label>Itérations max</label>
              <input type="number" step="1" name="max_iter" value="300">
            </div>
          </div>
        </fieldset>

        <button type="submit">Lancer l'optimisation</button>
      </form>
    </div></body></html>
    """
end

function results_html(res, p::Dict)
    n_iter_str = res.n_iter === nothing ? "n/a" : string(res.n_iter)
    msg_str = res.message === nothing ? "" : string(res.message)
    """
    <!doctype html><html lang="fr"><head><meta charset="utf-8">
    <title>Résultats — GRAPE</title>$PAGE_STYLE</head>
    <body><div class="wrap">
      <h1>Résultats de l'optimisation</h1>
      <p class="sub">$(msg_str)</p>

      <table class="results">
        <tr><td>J_T final</td><td>$(round(res.J_T, sigdigits=6))</td></tr>
        <tr><td>J_a final</td><td>$(round(res.J_a, sigdigits=6))</td></tr>
        <tr><td>Fidélité |⟨ψ1|φ(T)⟩|²</td><td>$(round(res.fidelity, sigdigits=6))</td></tr>
        <tr><td>Erreur ‖φ(T) − e^{iα}ψ1‖</td><td>$(round(res.error, sigdigits=6))</td></tr>
        <tr><td>Écart à la distance de Fubini-Study</td><td>$(round(res.fs_diff, sigdigits=6))</td></tr>
        <tr><td>Itérations effectuées</td><td>$(n_iter_str)</td></tr>
      </table>

      <div class="figure"><h3>1. Évolution des populations (contrôles optimisés)</h3>
        <img src="$(plot_to_data_uri(res.plots[1]))"></div>
      <div class="figure"><h3>2. Géodésique de Fubini-Study</h3>
        <img src="$(plot_to_data_uri(res.plots[2]))"></div>
      <div class="figure"><h3>3. Contrôles optimisés</h3>
        <img src="$(plot_to_data_uri(res.plots[3]))"></div>

      <a class="btn" href="/">← Nouveau calcul</a>
    </div></body></html>
    """
end

function error_html(e)
    """
    <!doctype html><html lang="fr"><head><meta charset="utf-8">
    <title>Erreur — GRAPE</title>$PAGE_STYLE</head>
    <body><div class="wrap">
      <h1>Une erreur est survenue</h1>
      <div class="err"><pre>$(sprint(showerror, e))</pre></div>
      <a class="btn" href="/">← Revenir au formulaire</a>
    </div></body></html>
    """
end

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

route("/") do
    html(form_html())
end

route("/run", method = POST) do
    p = parse_params()
    try
        res = run_optimization(p)
        html(results_html(res, p))
    catch e
        html(error_html(e))
    end
end

route("/health") do
    "OK"
end

# ------------------------------------------------------------------
# Démarrage du serveur (Render fournit le port via $PORT)
# ------------------------------------------------------------------

Genie.config.run_as_server = true
const PORT = parse(Int, get(ENV, "PORT", "8000"))
up(PORT, "0.0.0.0"; async = false)
