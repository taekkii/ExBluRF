import sqlite3
import os
import argparse

import numpy as np

def read_db(db_path, focal=None):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute("SELECT * FROM cameras")
    cameras_tuples = c.fetchall()

    if focal is not None:
        try:
            c.execute("UPDATE cameras SET prior_focal_length =1")
            
            params = np.frombuffer(cameras_tuples[0][4]).copy()
            params[0]=focal
            # params = params.tobytes()
            
            c.execute(f"UPDATE cameras SET params =?", [sqlite3.Binary(params)])
            conn.commit()
            print("updated focal")
        except Exception as e:
            print(e)
            exit()

    
    c.execute("SELECT * FROM images")
    images_tuples = c.fetchall()

    c.execute("SELECT * FROM cameras")
    cameras_tuples = c.fetchall()
    print([ele[0] for ele in c.description])

    return cameras_tuples, images_tuples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default="./test/000/database.db")
    parser.add_argument("--focal", type=float, default=None)
    # 88.88888249550146
    args = parser.parse_args()

    db_path = args.db_path
    
    cameras, _ = read_db(db_path, args.focal)
    print(cameras)
    