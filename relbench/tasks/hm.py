import duckdb
import pandas as pd

from relbench.data import Database, RelBenchLinkTask, RelBenchNodeTask, Table
from relbench.data.task_base import TaskType
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
    roc_auc,
)


class RecommendationTask(RelBenchLinkTask):
    r"""Predict the list of articles each customer will purchase in the next
    seven days"""

    name = "rel-hm-rec"
    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "customer_id"
    src_entity_table = "customer"
    dst_entity_col = "article_id"
    dst_entity_table = "article"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=7)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 12

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        # product = db.table_dict["product"].df
        customer = db.table_dict["customer"].df
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                transactions.customer_id,
                LIST(DISTINCT transactions.article_id) AS article_id
            FROM
                timestamp_df t
            LEFT JOIN
                transactions
            ON
                transactions.t_dat > t.timestamp AND
                transactions.t_dat <= t.timestamp + INTERVAL '{self.timedelta} days'
            GROUP BY
                t.timestamp,
                transactions.customer_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )


class ChurnTask(RelBenchNodeTask):
    r"""Predict the list of articles each customer will purchase in the next
    seven days"""

    name = "rel-hm-churn"
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "customer_id"
    entity_table = "customer"
    target_col = "churn"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=7)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        # product = db.table_dict["product"].df
        customer = db.table_dict["customer"].df
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                timestamp,
                customer_id,
                CAST(
                    NOT EXISTS (
                        SELECT 1
                        FROM transactions
                        WHERE
                            transactions.customer_id = customer.customer_id AND
                            transactions.t_dat > timestamp AND
                            transactions.t_dat <= timestamp + INTERVAL '{self.timedelta}'
                    ) AS INTEGER
                ) AS churn
            FROM
                timestamp_df,
                customer,
            WHERE
                EXISTS (
                    SELECT 1
                    FROM transactions
                    WHERE
                        transactions.customer_id = customer.customer_id AND
                        transactions.t_dat > timestamp - INTERVAL '{self.timedelta}' AND
                        transactions.t_dat <= timestamp
                )
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )
