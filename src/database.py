"""SQLAlchemy database resource for saving predictions"""

from contextlib import contextmanager

import pandas as pd
import sqlalchemy
from dagster import StringSource, resource
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


@resource(config_schema={"conn_str": StringSource})
def sqlalchemy_postgres_warehouse_resource(init_context):
    return SqlAlchemyPostgresWarehouse(init_context.resource_config["conn_str"])


class SqlAlchemyPostgresWarehouse:
    def __init__(self, conn_str):
        self._conn_str = conn_str
        self._engine = sqlalchemy.create_engine(self._conn_str)
        self.base = declarative_base(bind=self._engine)
        Session = sessionmaker(bind=self._engine)
        self.session = Session()

    @contextmanager
    def session_scope(self):
        Session = sessionmaker(bind=self._engine)
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
