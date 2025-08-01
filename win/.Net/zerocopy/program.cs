// C#
using System;
using System.Runtime.InteropServices;

public static class NativeMethods
{
    [DllImport("example.dll")]
    public static extern void ProcessDataZeroCopy(IntPtr data, int size);
    [DllImport("example.dll")]
    public static extern IntPtr CreateArray(out int length);
    [DllImport("example.dll")]
    public static extern void DestroyArray(IntPtr data);
}

public class MyProcessor
{
    static void ProcessMyData(byte[] myData)
    {
        unsafe
        {
            // Pin the managed array to get a stable pointer
            fixed (byte* ptr = myData)
            {
                Console.WriteLine("C++ processing:");
                NativeMethods.ProcessDataZeroCopy((IntPtr)ptr, myData.Length);
            }
            int length;
            IntPtr ptrAlloc = NativeMethods.CreateArray(out length);
            // Zerocopy
            Span<byte> span = new Span<byte>((void*)ptrAlloc, length);
            Console.WriteLine("C++ allocated content:");
            foreach (var v in span)
                Console.Write(v + " ");
            NativeMethods.DestroyArray(ptrAlloc);
        }
    }
    static void Main(string[] args) {
      byte[] data = new byte[] { 1, 2, 3, 4, 5 };
      ProcessMyData(data);
    }
}
