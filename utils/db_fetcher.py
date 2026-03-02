# utils/db_fetcher.py

import mysql.connector
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# DB_CONFIG = {
#     "host": "localhost",
#     "user": "root",
#     "password": "Linod+/+Mooxy/2021",
#     "database": "chatbot_db"
# }
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "database": "chatbot_db"
}

def get_raw_text_from_db(domain: str) -> str:
    """
    Fetch all 'answer' values for a given domain from the qa_pairs table.
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        table_name = f"data_{domain}"

        query = f"SELECT description from {table_name}"
        cursor.execute(query)
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        if not rows:
            logger.warning(f"No records found in DB for domain '{domain}'")

        return "\n".join(row["description"] for row in rows)

    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database access failed")
