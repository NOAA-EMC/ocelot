
from datetime import datetime

class Reader:
    def __init__(self, data_path: str):
        pass

    def create_data(start_date: datetime,
                    end_date: datetime,
                    data_type: str,
                    suffix: str = None,
                    append: bool = True) -> None:
        assert False, "This method should be implemented in a subclass."

class ReaderFactory:
    @staticmethod
    def create_reader(data_type: str, data_path: str):
        if data_type == 'bufr':
            from .bufr_pca_reader import Reader
            return Reader(data_path)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

