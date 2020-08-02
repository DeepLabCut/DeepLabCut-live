### DeepLabCut-live Processors

The `Processor` class allows users to implement processing or computation steps after DeepLabCut pose estimation. For example, a `Processor` can detect certain features of a pose and turn on an LED or an optogenetics laser, a `Processor` can implement a forward-prediction model that predicts animal's pose ~10-100 ms into the future to apply feedback with zero latency, or a `Processor` can do both.

The `Processor` is designed to be extremely flexible: it must only contain two methods: `Processor.process`, whose input and output is a pose as a numpy array, and `Processor.save`, which allows users to implement a method that saves any data the `Processor` acquires, such as the time that desired behavior occured or the times an LED or laser was turned on/off. The save method must be written by the user, so users can choose whether this data is saved as a text/csv, numpy, pickle, or pandas file to provide a few examples.

To write your own custom `Processor`, your class must inherit the base `Processor` class (see [here](./processor.py)):
```
from dlclive import Processor
class MyCustomProcessor(Processor):
    ...
```

To implement your processing steps, overwrite the `Processor.process` method:
```
    def process(pose, **kwargs):
        # my processing steps go here
        return pose
```

For example `Processor` objects that communicate with Teensy microcontrollers to [control an optogenetics laser](../../example_processors/TeensyLaser), [turn on an LED when upond detecting a mouse licking](../../example_processors/MouseLickLED), or [turn on an LED upon detecting a dog's rearing movement](../../example_processors/DogJumpLED), see our [example_teensy](../../example_processors) directory.
