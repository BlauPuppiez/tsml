/* 
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package utilities.class_counts;

import java.io.Serializable;
import java.util.Collection;
import java.util.Set;

/**
 * This is used by Aarons shapelet code and is down for depreciation
 * @author raj09hxu
 */
public abstract class ClassCounts implements Serializable {
    public abstract int get(double classValue);
    public abstract int get(int accessValue);
    public abstract void put(double classValue, int value);
    public abstract int size();
    public abstract Set<Double> keySet();
    public abstract Collection<Integer> values();
    public abstract void addTo(double classValue, int val);
}
