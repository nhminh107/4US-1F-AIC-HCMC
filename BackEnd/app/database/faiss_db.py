import json
from pathlib import Path
from tempfile import TemporaryDirectory

import faiss
import numpy as np


class FAISS_Manager:
    def __init__(
        self,
        dim: int,
        database_dir: str | Path | None = None,
    ):
        self.dim = dim

        resolved_database_dir = (
            Path(database_dir)
            if database_dir is not None
            else Path(__file__).resolve().parent
        )
        self.storage_path = resolved_database_dir / "storage"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.metadata_path = resolved_database_dir / "metadata.json"
        if not self.metadata_path.exists() or self.metadata_path.stat().st_size == 0:
            self.metadata_path.write_text("{}", encoding="utf-8")

        self.img_db = faiss.IndexFlatIP(dim)
        self.img_idx = self.__load_index("img.faiss", self.img_db)

        self.used_ids = self.__get_used_ids()
        self.current_id = max(self.used_ids, default=-1) + 1

    def __load_index(self, file_name: str, index):
        index_path = self.storage_path / file_name
        if not index_path.exists():
            return faiss.IndexIDMap2(index)

        loaded_index = faiss.read_index(str(index_path))
        if loaded_index.d != self.dim:
            raise ValueError(
                f"Index {file_name} has dim={loaded_index.d}, expected dim={self.dim}"
            )
        if not hasattr(loaded_index, "id_map"):
            raise ValueError(
                f"Index {file_name} does not contain explicit FAISS IDs"
            )
        return loaded_index

    def __read_metadata(self) -> dict:
        try:
            with self.metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a JSON object")
        return metadata

    def __get_used_ids(self) -> set[int]:
        metadata = self.__read_metadata()
        used_ids = {int(faiss_id) for faiss_id in metadata}

        if self.img_idx.ntotal > 0:
            used_ids.update(
                int(faiss_id)
                for faiss_id in faiss.vector_to_array(self.img_idx.id_map)
            )

        return used_ids

    def __prepare_embeddings(self, image_embeddings) -> np.ndarray:
        if hasattr(image_embeddings, "detach"):
            image_embeddings = image_embeddings.detach().cpu().float().numpy()
        elif hasattr(image_embeddings, "numpy"):
            image_embeddings = image_embeddings.numpy()

        image_embeddings = np.asarray(image_embeddings, dtype=np.float32)
        if image_embeddings.ndim == 1:
            image_embeddings = image_embeddings.reshape(1, -1)
        if (
            image_embeddings.ndim != 2
            or image_embeddings.shape[0] == 0
            or image_embeddings.shape[1] != self.dim
        ):
            raise ValueError(
                "Image embeddings must have shape "
                f"({self.dim},) or (batch_size, {self.dim})"
            )

        return np.ascontiguousarray(image_embeddings)

    def __validate_faiss_id(self, faiss_id: int) -> int:
        if isinstance(faiss_id, bool) or not isinstance(faiss_id, (int, np.integer)):
            raise TypeError("Each faiss_id must be an integer")

        validated_id = int(faiss_id)
        if validated_id < 0:
            raise ValueError("Each faiss_id must be greater than or equal to 0")
        if validated_id > np.iinfo(np.int64).max:
            raise ValueError("A faiss_id exceeds the int64 range")

        return validated_id

    def __resolve_faiss_ids(
        self,
        faiss_ids,
        batch_size: int,
    ) -> np.ndarray:
        if faiss_ids is None:
            resolved_ids = []
            candidate_id = self.current_id

            while len(resolved_ids) < batch_size:
                if candidate_id not in self.used_ids:
                    resolved_ids.append(candidate_id)
                candidate_id += 1
        else:
            try:
                provided_ids = list(faiss_ids)
            except TypeError as error:
                raise TypeError("faiss_ids must be an iterable of integers") from error

            if len(provided_ids) != batch_size:
                raise ValueError(
                    "The number of faiss_ids must match the number of embeddings"
                )
            resolved_ids = [
                self.__validate_faiss_id(faiss_id)
                for faiss_id in provided_ids
            ]

        if len(set(resolved_ids)) != len(resolved_ids):
            raise ValueError("faiss_ids must be unique within the batch")

        duplicated_ids = set(resolved_ids) & self.used_ids
        if duplicated_ids:
            duplicated_id = min(duplicated_ids)
            raise ValueError(f"faiss_id {duplicated_id} already exists")

        return np.asarray(resolved_ids, dtype=np.int64)

    def __add_metadata(
        self,
        faiss_ids: np.ndarray,
        img_names: list[str],
    ) -> None:
        metadata = self.__read_metadata()
        for faiss_id, img_name in zip(faiss_ids.tolist(), img_names):
            metadata[str(faiss_id)] = {
                "idx": faiss_id,
                "img_name": img_name,
            }

        with self.metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=4)

    def add_image_embedding(
        self,
        image_embedding,
        img_name: str,
        faiss_id: int | None = None,
    ) -> int:
        """Add one image embedding and return its FAISS ID."""
        faiss_ids = None if faiss_id is None else [faiss_id]
        added_ids = self.add_image_embeddings(
            image_embeddings=image_embedding,
            img_names=[img_name],
            faiss_ids=faiss_ids,
        )
        return added_ids[0]

    def add_image_embeddings(
        self,
        image_embeddings,
        img_names: list[str],
        faiss_ids=None,
    ) -> list[int]:
        """Add a batch of image embeddings and return their FAISS IDs."""
        prepared_embeddings = self.__prepare_embeddings(image_embeddings)
        batch_size = prepared_embeddings.shape[0]

        if len(img_names) != batch_size:
            raise ValueError(
                "The number of img_names must match the number of embeddings"
            )
        if not all(isinstance(img_name, str) for img_name in img_names):
            raise TypeError("Each img_name must be a string")

        resolved_ids = self.__resolve_faiss_ids(faiss_ids, batch_size)
        faiss.normalize_L2(prepared_embeddings)

        self.img_idx.add_with_ids(prepared_embeddings, resolved_ids)
        self.__add_metadata(resolved_ids, img_names)

        added_ids = resolved_ids.tolist()
        self.used_ids.update(added_ids)
        self.current_id = max(self.current_id, max(added_ids) + 1)
        return added_ids

    def save_indexes(self) -> None:
        """Persist the image index to storage/img.faiss."""
        faiss.write_index(
            self.img_idx,
            str(self.storage_path / "img.faiss"),
        )

    def search_img(
        self,
        query_vector,
        top_k: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search the image index and return distance and FAISS ID arrays."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query = self.__prepare_embeddings(query_vector)
        if query.shape[0] != 1:
            raise ValueError("search_img accepts exactly one query vector")
        faiss.normalize_L2(query)

        result_count = min(top_k, self.img_idx.ntotal)
        if result_count == 0:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int64),
            )

        distances, ids = self.img_idx.search(query, result_count)
        return distances[0], ids[0]

    def get_metadata_by_id(self, faiss_id: int) -> dict | None:
        """Return metadata for one FAISS ID."""
        metadata = self.__read_metadata()
        return metadata.get(str(faiss_id))

    def get_all_metadata(self) -> dict:
        """Return all image metadata."""
        return self.__read_metadata()


def main() -> None:
    """Run a batch insertion example without modifying the real database."""
    image_embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    img_names = ["image_001.jpg", "image_002.jpg", "image_003.jpg"]
    faiss_ids = [101, 102, 103]

    with TemporaryDirectory() as temp_dir:
        manager = FAISS_Manager(dim=4, database_dir=temp_dir)
        added_ids = manager.add_image_embeddings(
            image_embeddings=image_embeddings,
            img_names=img_names,
            faiss_ids=faiss_ids,
        )
        manager.save_indexes()

        distances, result_ids = manager.search_img(
            image_embeddings[0],
            top_k=3,
        )

        print(f"Added FAISS IDs: {added_ids}")
        print(f"Search result IDs: {result_ids.tolist()}")
        print(f"Search distances: {distances.tolist()}")
        print(f"Metadata: {manager.get_all_metadata()}")
        print(f"Temporary index: {manager.storage_path / 'img.faiss'}")


if __name__ == "__main__":
    main()
