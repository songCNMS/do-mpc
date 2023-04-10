from django.db import models

# Create your models here.

class SimulatorParameterModel(models.Model):
        # fields of the model
    parameter_name = models.CharField(max_length = 200)
    parameter_value = models.CharField(max_length = 20)
    parameter_type = models.CharField(max_length = 10)
        # renames the instances of the model
        # with their title name
    def __str__(self):
        return self.parameter_name