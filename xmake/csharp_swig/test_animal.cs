using System;
using static AnimalModule;

class Program
{
    static Span<byte> AllocateSpan(int size)
    {
      return new Span<byte>(new byte[size]);
    }
    static void Main()
    {
        using var animal = new Animal("rabbit");
        Console.WriteLine("Animal name: " + animal.GetName());
        animal.Walk();
    }
}