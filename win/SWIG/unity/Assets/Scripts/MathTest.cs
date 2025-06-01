using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using static MathLibrary;

public class MathTest : MonoBehaviour
{
    void Start()
    {
        int val = MathLibrary.add(15, 27);
        Debug.Log($"C++ add: 15 + 27 = {val}");
        val = MathLibrary.subtract(15, 27);
        Debug.Log($"C++ substract: 15 - 27 = {val}");
    }

    void Update()
    {
        
    }
}
