from BackEnd.app.database.postgre_db import Postgre_Manager

db = Postgre_Manager()

# res = db.add_video(
#     "001", 30.0, 1500, "test01"
# )
# res2 = db.add_scene(
#     "001", "001", 100, 200
# )

# res3 = db.add_scene(
#     "002", "001", 100, 200
# )

res4 = db.add_video(
    "002", 31.5, 1000, "test02"
)