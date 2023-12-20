// linear n-dimensional quadtree / octtree

/*
	var tree = new NDTree(2, [ 0, 0, 100, 100 ]);
	tree.insert(id, [ 40, 40, 10, 10 ]);

	var query = tree.query([ 10, 10, 50, 50 ]);
	while (query.next()) console.log(query.id() + ' => ' + query.data());
*/

function NDTree(dims, bbox, depth, leafs)
{
	this.dims = dims;
	this.stride = this.dims * 2 + 3;
	this.branches = Math.pow(2, this.dims);
	this.maxdepth = depth || 3;
	this.maxleafs = leafs || 64;
	this.maxsize = 0;
	this.margin = 0;
	this.leafs = 0;
	this.size = 0;
	this.maskbits = 0;
	this.bbox = new Array(this.dims * 2);
	this.flip = new Array(this.dims);
	this.iterator = new QueryIterator();

	for (var i = 0; i < this.maxdepth; i++)
	{
		this.maxsize += Math.pow(this.branches, (i + 1));
	}

	this.maxsize += 1;
	this.maxsize += this.maxleafs;
	this.maxsize *= this.stride;
	this.data = new Array(this.maxsize);

	this.X = 0;
	this.Y = 1;
	this.Z = 2;
	this.W = this.dims + 0;
	this.H = this.dims + 1;
	this.D = this.dims + 2;
	this.DEPTH = this.dims * 2;
	this.ID = this.DEPTH; // leaf id is stored in the same place as depth
	this.NODE = this.DEPTH + 1;
	this.LEAF = this.DEPTH + 2;

	this.alloc(bbox, 0);
}

NDTree.prototype.mask = function(node, bbox)
{
	var mask = 0;
	var i = 0;
	var j = 0;
	var dim = 1;
	var set = 1;
	var flag = 1;
	var center;

	this.maskbits = 0;

	while (i < this.dims)
	{
		center = this.data[node + i] + ((this.data[node + this.dims + i] | 0) >> 1);
		this.bbox[i] = ((bbox[i] - this.margin) < center);
		this.bbox[this.dims + i] = ((bbox[i] + bbox[this.dims + i] + this.margin) > center);
		this.flip[i] = true;
		i++;
	}

	for (i = 1; i <= this.branches; i++)
	{
		j = 0;
		dim = 1;
		set = 1;

		while (j < this.dims)
		{
			set *= this.bbox[this.dims * this.flip[j] + j];
			if (i % dim == 0) this.flip[j] = ! this.flip[j];
			dim *= 2;
			j++;
		}

		this.maskbits += set;
		mask = mask | (flag * set);
		flag *= 2;
	}

	return mask;
};

NDTree.prototype.overlap = function(node, bbox)
{
	var i = 0;

	while (i < this.dims)
	{
		if ((this.data[node + i] + this.data[node + this.dims + i] < bbox[i]) || (this.data[node + i] > bbox[i] + bbox[this.dims + i])) return false;
		i++;
	}

	return true;
};

NDTree.prototype.clear = function()
{
	this.size = 0;
	this.leafs = 0;
	this.data[this.NODE] = -1;
	this.data[this.LEAF] = -1;
};

NDTree.prototype.alloc = function(bbox, depth)
{
	if (this.size >= this.maxsize) return -1;

	var i = 0;

	while (i < this.dims)
	{
		this.data[this.size + i] = bbox[i];
		this.data[this.size + this.dims + i] = bbox[this.dims + i];
		i++;
	}

	this.data[this.size + this.DEPTH] = depth;
	this.data[this.size + this.NODE] = -1;
	this.data[this.size + this.LEAF] = -1;

	this.size += this.stride;

	return this.size - this.stride;
};

NDTree.prototype.split = function(node)
{
	node = node || 0;

	if (node < 0 || node >= this.maxsize) return -1;
	if (this.data[node + this.NODE] != -1) return this.data[node + this.NODE];
	if (this.data[node + this.DEPTH] >= this.maxdepth) return -1;

	this.data[node + this.NODE] = this.size;

	var i = 0;
	var j = 0;
	var dim = 1;

	while (i < this.dims)
	{
		this.bbox[this.dims + i] = (this.data[node + this.dims + i] | 0) >> 1;
		this.flip[i] = true;
		i++;
	}

	for (i = 1; i <= this.branches; i++)
	{
		j = 0;
		dim = 1;

		while (j < this.dims)
		{
			this.bbox[j] = this.data[node + j] + this.bbox[this.dims + j] * this.flip[j];
			if (i % dim == 0) this.flip[j] = ! this.flip[j];
			dim *= 2;
			j++;
		}

		this.alloc(this.bbox, this.data[node + this.DEPTH] + 1);
	}

	return this.size;
};

NDTree.prototype.insert = function(id, bbox, node)
{
	node = node || 0;

	if (node < 0 || node >= this.maxsize || this.leafs >= this.maxleafs) return -1;

	var mask = this.mask(node, bbox);

	if (this.maskbits == 1 && this.split(node) != -1)
	{
		var i = 0;
		var flag = 1;
		var next = this.data[node + this.NODE];

		while (i < this.branches)
		{
			(mask & flag) && this.insert(id, bbox, next + i * this.stride);
			flag *= 2;
			i++;
		}

		return next;
	}

	var leaf = this.alloc(bbox, this.data[node + this.DEPTH]);
	var next = this.data[node + this.LEAF];

	this.data[leaf + this.ID] = id;
	this.data[node + this.LEAF] = leaf;
	this.data[leaf + this.LEAF] = next;
	this.leafs++;

	return leaf;
};

NDTree.prototype.query = function(bbox, node, depth)
{
	node = node || 0;
	depth = depth || 0;

	if (depth == 0) this.iterator.clear();
	if (node < 0 || node >= this.maxsize) return this.iterator;

	var next = this.data[node + this.LEAF];

	while (next != -1)
	{
		if (this.overlap(next, bbox))
		{
			var i = 0;
			var b = new Array(this.dims * 2);

			while (i < this.dims)
			{
				b[i] = this.data[next + i];
				b[this.dims + i] = this.data[next + this.dims + i++];
			}

			this.iterator.insert(this.data[next + this.ID], b);
		}

		next = this.data[next + this.LEAF];
	}

	next = this.data[node + this.NODE];

	if (next != -1)
	{
		var i = 0;
		var flag = 1;
		var mask = this.mask(node, bbox);

		while (i < this.branches)
		{
			(mask & flag) && this.query(bbox, next + i * this.stride, depth + 1);
			flag *= 2;
			i++;
		}
	}

	return this.iterator;
};

function QueryIterator()
{
	this.head = -1;
	this.size = 0;
	this._id = []
	this._data = [];
}

QueryIterator.prototype.clear = function()
{
	this.head = -1;
	this.size = 0;
};

QueryIterator.prototype.rewind = function()
{
	this.head = -1;
};

QueryIterator.prototype.empty = function()
{
	return this.size == 0;
};

QueryIterator.prototype.end = function()
{
	return this.head >= this.size;
};

QueryIterator.prototype.next = function()
{
	return this.head++ < this.size - 1;
};

QueryIterator.prototype.id = function()
{
	if (this.head == -1) this.head = 0;
	return this._id[this.head];
};

QueryIterator.prototype.data = function()
{
	if (this.head == -1) this.head = 0;
	return this._data[this.head];
};

QueryIterator.prototype.insert = function(id, data)
{
	if (this.size < this._data.length)
	{
		this._id[this.size] = id;
		this._data[this.size++] = data;
	} else
	{
		this._id.push(id);
		this._data.push(data);
		this.size++;
	}

	return this.size - 1;
};

QueryIterator.prototype.update = function(id, data, at)
{
	if (at >= 0 && at < this.size)
	{
		this._id[at] = id;
		this._data[at] = data;

		return at;
	}

	return this.insert(id, data);
};
