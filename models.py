from django.db import models


class DatabaseQuery(models.Model):
    query = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    api_used = models.CharField(max_length=50)

    class Meta:
        verbose_name_plural = "Database Queries"

    def __str__(self):
        return f"{self.query[:50]}... ({self.api_used})"
