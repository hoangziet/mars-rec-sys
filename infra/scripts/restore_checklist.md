# Restore Checklist

1. Restore PostgreSQL service and database dump.
2. Restore MinIO service and artifact bucket contents.
3. Restore server-side env/config files.
4. Start MLflow with the expected backend/artifact configuration.
5. Start Nginx with Basic Auth enabled.
6. Verify SSH tunnel access to `http://127.0.0.1:8080`.
7. Open MLflow UI and confirm experiments are visible.
8. Download one known checkpoint artifact and load it with project code.
