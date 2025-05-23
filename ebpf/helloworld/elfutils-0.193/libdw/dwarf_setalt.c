/* Provides the data referenced by the .gnu_debugaltlink section.
   Copyright (C) 2014, 2018 Red Hat, Inc.
   This file is part of elfutils.

   This file is free software; you can redistribute it and/or modify
   it under the terms of either

     * the GNU Lesser General Public License as published by the Free
       Software Foundation; either version 3 of the License, or (at
       your option) any later version

   or

     * the GNU General Public License as published by the Free
       Software Foundation; either version 2 of the License, or (at
       your option) any later version

   or both in parallel, as here.

   elfutils is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received copies of the GNU General Public License and
   the GNU Lesser General Public License along with this program.  If
   not, see <http://www.gnu.org/licenses/>.  */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include "libdwP.h"

void
dwarf_setalt (Dwarf *main, Dwarf *alt)
{
  mutex_lock (main->dwarf_lock);

  if (main->alt_fd != -1)
    {
      INTUSE(dwarf_end) (main->alt_dwarf);
      close (main->alt_fd);
      main->alt_fd = -1;
    }

  main->alt_dwarf = alt;
  mutex_unlock (main->dwarf_lock);
}
INTDEF (dwarf_setalt)
