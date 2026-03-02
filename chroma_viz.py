from __future__ import annotations

import argparse
import csv
from pathlib import Path

import chromadb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _reduce_embeddings(embeddings: np.ndarray, method: str, dims: int) -> np.ndarray:
    if method == "pca":
        reducer = PCA(n_components=dims, random_state=42)
        return reducer.fit_transform(embeddings)

    perplexity = max(2, min(30, embeddings.shape[0] - 1))
    reducer = TSNE(
        n_components=dims,
        random_state=42,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    return reducer.fit_transform(embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ChromaDB embeddings in 2D or 3D")
    parser.add_argument("--db-path", default=".chroma/indictment_db", help="Path to ChromaDB persistent directory")
    parser.add_argument("--collection", default="rag_tool_collection", help="Collection name")
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca", help="Dimensionality reduction method")
    parser.add_argument("--dims", choices=[2, 3], type=int, default=3, help="Output dimensions for visualization")
    parser.add_argument("--limit", type=int, default=500, help="Max number of points to plot")
    parser.add_argument("--output", default=".chroma/chroma_viz_3d.png", help="Path to static output image")
    parser.add_argument("--csv", default=".chroma/chroma_viz_points_3d.csv", help="Path to exported points CSV")
    parser.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        default=True,
        help="Generate interactive HTML visualization (default: enabled)",
    )
    parser.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="Disable interactive HTML output",
    )
    parser.add_argument(
        "--html-output",
        default=".chroma/chroma_viz_interactive_3d.html",
        help="Path to interactive HTML output",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Chroma DB path not found: {db_path}")

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(args.collection)

    records = collection.get(limit=args.limit, include=["embeddings", "metadatas", "documents"])
    ids = records.get("ids") or []
    embeddings = records.get("embeddings")

    if not ids or embeddings is None:
        raise RuntimeError("No embeddings found in collection. Run indexing first.")

    matrix = np.asarray(embeddings, dtype=np.float32)
    points = _reduce_embeddings(matrix, args.method, args.dims)

    metadatas = records.get("metadatas") or [{} for _ in ids]
    documents = records.get("documents") or ["" for _ in ids]

    chunk_indices: list[int] = []
    labels: list[str] = []
    previews: list[str] = []
    for idx, metadata in enumerate(metadatas):
        metadata = metadata or {}
        chunk_index = metadata.get("chunk_index", idx)
        chunk_indices.append(int(chunk_index) if str(chunk_index).isdigit() else idx)
        labels.append(f"{ids[idx][:8]} | chunk={chunk_index}")
        previews.append((documents[idx] or "").replace("\n", " ")[:180])

    if args.interactive:
        try:
            import plotly.express as px
        except ImportError as error:
            raise RuntimeError("plotly is required for --interactive mode. Run: uv sync") from error

        html_output_path = Path(args.html_output)
        html_output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.dims == 3:
            fig = px.scatter_3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                color=chunk_indices,
                hover_name=labels,
                hover_data={
                    "chunk_index": chunk_indices,
                    "id": ids,
                    "preview": previews,
                },
                title=f"Chroma Embeddings ({args.collection}) - {args.method.upper()} 3D",
                color_continuous_scale="Plasma",
            )
        else:
            fig = px.scatter(
                x=points[:, 0],
                y=points[:, 1],
                color=chunk_indices,
                hover_name=labels,
                hover_data={
                    "chunk_index": chunk_indices,
                    "id": ids,
                    "preview": previews,
                },
                title=f"Chroma Embeddings ({args.collection}) - {args.method.upper()} 2D",
                color_continuous_scale="Plasma",
            )

        fig.update_traces(marker={"size": 6, "opacity": 0.85})
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1026",
            plot_bgcolor="#0b1026",
            font={"family": "Arial", "size": 13},
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
            coloraxis_colorbar={"title": "chunk_index"},
        )
        fig.write_html(str(html_output_path), include_plotlyjs="cdn")
        print(f"Saved interactive plot to: {html_output_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    if args.dims == 3:
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=chunk_indices,
            cmap="viridis",
            s=45,
            alpha=0.85,
            edgecolors="none",
        )
        ax.set_title(f"Chroma Embeddings ({args.collection}) - {args.method.upper()} 3D")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        for i, text in enumerate(labels[:20]):
            ax.text(points[i, 0], points[i, 1], points[i, 2], text, fontsize=6, alpha=0.75)
    else:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=chunk_indices,
            cmap="viridis",
            s=45,
            alpha=0.85,
            edgecolors="none",
        )
        ax.set_title(f"Chroma Embeddings ({args.collection}) - {args.method.upper()} 2D")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        for i, text in enumerate(labels[:30]):
            ax.annotate(text, (points[i, 0], points[i, 1]), fontsize=7, alpha=0.75)

    fig.colorbar(scatter, label="chunk_index")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        headers = ["id", "x", "y", "chunk_index", "document_preview"]
        if args.dims == 3:
            headers.insert(3, "z")
        writer.writerow(headers)
        for i, point in enumerate(points):
            preview = previews[i]
            if args.dims == 3:
                writer.writerow([ids[i], float(point[0]), float(point[1]), float(point[2]), chunk_indices[i], preview])
            else:
                writer.writerow([ids[i], float(point[0]), float(point[1]), chunk_indices[i], preview])

    print(f"Saved plot to: {output_path}")
    print(f"Saved points CSV to: {csv_path}")
    print(f"Plotted points: {len(ids)}")
    print(f"Dimensions: {args.dims}D")
    if args.interactive:
        print("How to read it: nearby points are semantically similar chunks; hover a point to inspect chunk id and text preview.")
        print("Tip: rotate (left mouse), pan (right mouse), and zoom (wheel) in the HTML view.")


if __name__ == "__main__":
    main()
