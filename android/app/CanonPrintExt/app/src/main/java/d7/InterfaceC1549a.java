package d7;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

/** Stub for Canon preference-binding annotation on CLSS structs. */
@Retention(RetentionPolicy.RUNTIME)
public @interface InterfaceC1549a {
    boolean defBoolean() default false;

    int defInt() default 65535;

    String key() default "";
}
