from ._anvil_designer import webappTemplate
from anvil import *
import anvil.server


class webapp(webappTemplate):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Any code you write here will run when the form opens.

  def file_loader_1_change(self, file, **event_args):
    guess,accuracy = anvil.server.call('classify_image',file)
    self.label_1.text = f'{guess} \nconfidence: {accuracy}%'
    self.image_1.source = file
    