from app.database.models import Base, KnowledgeChunk


print("Model loaded successfully!")
print("Table name:", KnowledgeChunk.__tablename__)
print("Tables registered with Base:")

for table_name in Base.metadata.tables:
    print("-", table_name)