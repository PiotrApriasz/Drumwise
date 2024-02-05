using Drumwise.Application.Entities;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Drumwise.Infrastructure.Data.Configurations;

public class UserRatingConfiguration : IEntityTypeConfiguration<UserRating>
{
    public void Configure(EntityTypeBuilder<UserRating> builder)
    {
        builder.Property(p => p.Comment)
            .IsRequired();

        builder.Property(p => p.Mark)
            .IsRequired();
        
        builder.Property(p => p.AssignedTo)
            .IsRequired();
    }
}