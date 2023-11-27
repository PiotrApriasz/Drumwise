using Microsoft.AspNetCore.Identity;

namespace Drumwise.Infrastructure.Identity;

public class ApplicationUser : IdentityUser
{
    public string? Name { get; set; }
    public string? Surname { get; set; }
    
    /// <summary>
    /// Years and fraction of year of experience on drums 
    /// </summary>
    public float Experience { get; set; }
}