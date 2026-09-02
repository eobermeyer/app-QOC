FROM julia:1.10

# Dépendances système utiles à la compilation / au backend GR de Plots.jl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# GR headless (pas de serveur X dans le conteneur)
ENV GKSwstype=100 \
    JULIA_DEPOT_PATH=/opt/julia_depot \
    JULIA_NUM_THREADS=1

WORKDIR /app
COPY app.jl /app/app.jl

# Installe les paquets dans l'environnement du projet et précompile
# (Pkg.add génère lui-même Project.toml / Manifest.toml : pas besoin de
# les fournir à la main, ce qui évite tout risque d'UUID erroné.)
RUN julia --project=/app -e '\
    using Pkg; \
    Pkg.add([ \
        "Genie", \
        "GRAPE", \
        "QuantumPropagators", \
        "QuantumControl", \
        "Plots", \
    ]); \
    Pkg.precompile()'

# Render fournit dynamiquement la variable PORT ; app.jl la lit elle-même.
EXPOSE 8000

CMD ["julia", "--project=/app", "-e", "include(\"/app/app.jl\")"]
